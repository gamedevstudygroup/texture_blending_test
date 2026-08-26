import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    from time import perf_counter

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from PIL import Image
    # These imports are kept inside the notebook so the file can be opened as a
    # standalone marimo app without requiring a separate support module.


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Network Simplex implementation
    mostly abandoned because it is simply too slow (even very optimized implementations) and uses too much memory for large textures.

    our version currently lacks ability to descend children and so has to recompute entire depth and potential each pivot

    for 100 pixels, this takes something like 30s, with 8000 pivots
    fasttransport observes time complexity of O(N2), ours is likely more like O(N3)
    both are unviable for textures (512x512 pixels minimum)
    the original high perf noise paper used 256x256 https://inria.hal.science/hal-01824773/document which took several minutes with fasttransport.
    """)
    return


@app.function
def gen_gaussian(size, *, mean=0.5, std=1 / 6, seed=None, levels=256):
    """Generate image with Gaussian distribution."""
    width, height = size
    if std <= 0.0 or levels < 2 or not 0.0 <= mean <= 1.0:
        raise ValueError("mean must be in [0, 1], std positive, and levels at least 2")

    # intensity_levels = np.linspace(0.0, 1.0, levels, dtype=np.float32)
    # probability = np.exp(-0.5 * ((intensity_levels - mean) / std) ** 2)
    # probability /= probability.sum()
    # return np.random.default_rng(seed).choice(
    #     intensity_levels, size=(height, width, 3), p=probability
    # )

    rng = np.random.default_rng(seed)
    array = rng.random((width, height, 3))

    # apply inverse cdf of gaussian distribution
    from scipy.special import erfinv
    return 1/6 * 2**.5 * erfinv( 2 * array - 1 ) + mean


@app.cell(hide_code=True)
def network_simplex():
    # we map every pixel in texture image to a pixel in the gaussian image
    # note: every *pixel*, not every pixel value: imagine there were lots of a certain value, it would throw off the resulting histogram
    # you can think of this as deciding where in the image to place the random gaussian pixels
    # ie. we rearrange the pixels of the gaussian image to look like the texture image, because this results in a similar image which is still gaussian.

    p = False

    # minimize total cost of all pixel mappings
    class NetworkSimplex():
        def __init__(self, image):    
            self.COST_SCALE = 10_000 # big enough costs differ, small enough that totals fit in int32
            # the above greatly effects the likelihood of the solution being unique

            self.artificial_edge_cost = 4 * self.COST_SCALE # larger than largest dist on floats.

            self.N = image.shape[0] * image.shape[0]

            self.image = image

            # resize everything to 1d for convenience
            self.source = image.copy()
            self.source.resize((self.N, 3))
            self.target = gen_gaussian(image.shape[:2], seed=1)
            self.target.resize((self.N, 3), refcheck=False)

            # network simplex needs a spanning tree for bookkeeping
            # it is best to think of this tree as very rooted, as in, we care about parents and root orientation and not so much about children (at least we don't traverse in that direction)
            # storing parent means we can quickly get the cycle
            # this is more of a undirected graph with direction metadata. when we say "cycle" we arent considering direction

            # all used (1) edges are in tree, but we need extra unused (0) edges to span all the source and target pixels

            # for the algo to work and avoid non-termination, an invariant is required on the tree
            # all zedges must point (source -> target) away from the root!

            # what we have is a graph, but constrained to source nodes pointing to target nodes
            # it feels weird to have source and target nodes just be nodes, but I cant think of a more specialized structure
            # target nodes are those with index >= N (top half of the arrays)

            self.parent = np.full(self.N * 2, -1, dtype=np.int32) # connect everything to an artificial root -1
            self.value  = np.ones(self.N * 2, dtype=np.int32) # the artificial root edges *are* "used" at the start
            self.depth  = np.ones(self.N * 2, dtype=np.int32) # store this so we can efficiently find the cycle

            # potential is a way of precomputing the effect of flipping all the 0/1s from root to this cell
            # defined by P(Xn) = P(Ym) - C(Xn->Ym) for each such edge in the basis
            # root has potential 0, so incomming edges are -C and outgoing are C
            # NOTE: subtracting potentials will cancel out everything above the common ancestor 
            self.potential = np.ones(self.N*2, dtype=np.int32) * self.artificial_edge_cost
            self.potential[0:self.N] *= -1 # X->R negative, R->Y positive

            self.total_potential = 0
            self.total_cost = self.artificial_edge_cost * self.N * 2
            print(f"initial_cost {self.total_cost}")

            self.list = list() # scratchpad

        def sort_initial_by_nearest(self):
            taken = np.zeros(self.N)
            self.total_cost = 0
            for x in range(self.N):
                # min cost 
                min_cost = None
                min_y = None
                for y in range(self.N):
                    if not taken[y]:
                        cost = self.cost(x, y+self.N)
                        if min_cost is None or cost < min_cost:
                            min_cost = cost
                            min_y = y

                # map x -> min_y
                assert min_y is not None
                taken[min_y] = 1
                yi = min_y + self.N
                self.parent[x] = yi
                self.value[x] = 1
                self.parent[yi] = -1
                self.value[yi] = 0

                self.total_cost += min_cost
            assert np.all(taken)

            self.recompute_depth_and_potential()
            self.verify_invariants()
            print(f"initial_cost {self.total_cost}")

        def index_helper(self, i):
            p = self.parent[i]
            if i < self.N:
                return i, p
            else:
                return p, i


        def verify_totals(self):
            # make sure our logic is right by summing potential and cost and comparing to running totals 
            assert self.total_potential == np.sum(self.potential), (np.sum(self.potential), self.total_potential)

            total = 0
            for i in range(self.N*2):
                if self.value[i] == 1:
                    s_i, t_i = self.index_helper(i)
                    total += self.cost(s_i, t_i)
            assert self.total_cost == total, (self.total_cost, total)

        def verify_invariants(self):
            self.verify_totals()
            for i in range(self.N * 2):
                if self.value[i] == 0:
                    assert i >= self.N, "zero edges must point away from root Y<-0-X->...R"

                parent = self.parent[i]
                if i >= self.N: # is a Y<-X...R 
                    if parent == -1:
                        assert self.potential[i] == self.cost(parent, i)
                    else:
                        assert self.potential[i] == self.potential[parent] + self.cost(parent, i)
                if i < self.N: # is a X->Y...R 
                    if parent == -1:
                        assert self.potential[i] == -self.cost(i, parent)
                    else: 
                        assert self.potential[parent] == self.potential[i] + self.cost(i, parent)

        def get_map(self):
            map = np.full(self.N, -1, dtype=np.int32)

            for i in range(self.N * 2):
                if self.value[i] == 1:
                    parent = self.parent[i]
                    assert parent != -1
                    if i < self.N:
                        assert parent >= self.N
                        assert map[i] == -1
                        map[i] = parent - self.N
                    else:
                        assert map[parent] == -1
                        assert i >= self.N
                        map[parent] = i - self.N
        
            return map

        def faster_solve(self):
            self.verify_invariants()

            # picks first valid edge to add
            count = 0
            found = True
            while found:
                found = False
                for x in range(self.N):
                    # compute all reduced costs in this block

                    offset = 0
                    C = 100
                    reduced = None
                    def comp():
                        nonlocal reduced
                        src = self.source[x]                  # (3,)
                        dst = self.target[offset: min(offset+C, self.N) ]                     # (N, 3)
        
                        cost = np.sum((dst - src) ** 2, axis=1)
                        cost = (cost * self.COST_SCALE).astype(np.int32)
        
                        reduced = (
                            cost
                            + self.potential[x]
                            - self.potential[self.N + offset: self.N + offset + len(dst)]
                        )
        
                    comp()

                    for y in range(self.N, self.N*2):
                        if y - self.N - offset >= len(reduced):
                            offset = y - self.N
                            comp()
                        r = reduced[y - self.N - offset]
                        #assert r == self.reduced_cost(x,y)
                        if r < 0:
                            found = True
                            count += 1
                            self.incomming_edge(x, y)
                            self.recompute_depth_and_potential()
                            self.verify_invariants()

                            offset = y - self.N + 1
                            comp()
            
            

                print(f"{count} pivots, cost={self.total_cost} p={self.total_potential}")

            self.map = self.get_map()

        def simple_solve(self):
            self.verify_invariants()

            # picks first valid edge to add
            count = 0
            found = True
            while found:
                found = False
                for x in range(self.N):
                    for y in range(self.N, self.N*2):
                        if self.reduced_cost(x,y) < 0:
                            found = True
                            count += 1
                            self.incomming_edge(x, y)
                            self.recompute_depth_and_potential()
                            self.verify_invariants()

                print(f"{count} pivots, cost={self.total_cost} p={self.total_potential}")

            self.map = self.get_map()

        def recompute_depth_and_potential(self):
            """Rebuild depths and signed potentials from the parent tree."""
            node_count = self.N * 2
            depths = np.full(node_count, -1, dtype=np.int32)
            potentials = np.empty(node_count, dtype=self.potential.dtype)

            for start in range(node_count):
                if depths[start] != -1:
                    continue

                path = []
                seen = set()
                node = start
                # Walk to the artificial root or an already resolved ancestor.
                while depths[node] == -1:
                    assert node not in seen, f"cycle in parent tree at node {node}"
                    seen.add(node)
                    path.append(node)
                    parent = int(self.parent[node])
                    if parent == -1:
                        break
                    assert 0 <= parent < node_count, (node, parent)
                    node = parent

                # Resolve back down the path so every parent is ready first.
                while path:
                    node = path.pop()
                    parent = int(self.parent[node])
                    if parent == -1:
                        depths[node] = 1
                        potentials[node] = (
                            -self.cost(node, parent)
                            if node < self.N
                            else self.cost(parent, node)
                        )
                    else:
                        depths[node] = depths[parent] + 1
                        potentials[node] = (
                            potentials[parent] - self.cost(node, parent)
                            if node < self.N
                            else potentials[parent] + self.cost(parent, node)
                        )

            self.depth[:] = depths
            self.potential[:] = potentials
            self.total_potential = int(np.sum(potentials, dtype=np.int64))

        def incomming_edge(self, s_i, t_i):
            if p:
                print("bonk")
            # check if degen
            # inserting an edge as used means that the 
            # in order to insert edge with flow=1 (used) we must reduce flow for connected edges (make them unused) and continue this pattern around the cycle

            assert t_i >= self.N

            s_d = self.depth[s_i]
            t_d = self.depth[t_i]
            cost = self.cost(s_i, t_i)

            # see comment in reduced_cost function
            reduced_cost = cost + self.potential[s_i] - self.potential[t_i] 

            assert reduced_cost < 0 # we should only be inserting if reduced cost is negative

            # walk up both sides of incomming edge to find common ancestor
            def walk_up():
                s_dd = s_d 
                s_ii = s_i
                while s_dd > t_d:
                    yield s_ii, s_d - s_dd, None
                    s_dd -= 1 
                    s_ii = self.parent[s_ii]

                t_dd = t_d 
                t_ii = t_i
                while t_dd > s_d:
                    yield t_ii, None, t_d - t_dd
                    t_dd -= 1 
                    t_ii = self.parent[t_ii]

                # we have now lined up depths
                assert t_dd == s_dd
                d = t_dd

                while t_ii != s_ii:
                    yield s_ii, s_d - d, None
                    yield t_ii, None, t_d - d
                    d -= 1
                    t_ii = self.parent[t_ii]
                    s_ii = self.parent[s_ii]

                yield t_ii, s_d - d, t_d - d # common ancestor

            # when checking degen only odd edges matter
            degen = False
            common_ancestor = None
            exiting_edge = None
            for ii, s_side, t_side in walk_up():
                if p:
                    print(s_i, t_i, ii, s_side, t_side)
                assert ii is not None
                if s_side is not None and t_side is not None:
                    common_ancestor = ii
                    break # this is common ancestor node, all edges have been checked
                index = t_side if s_side is None else s_side

                assert ii != -1

                # index equals 0 for first edge connected to incomming source or target, and increases up the tree
                # meaning that even index nodes are the ones that must get subtracted from (become unused), so if any are zero (already unused), this insertion is degen.
                if index % 2 == 0 and self.value[ii] == 0:
                    assert s_side is None, "source side cannot have zeros without breaking invariant"
                    if not degen: 
                        exiting_edge = ii
                        degen = True

            assert common_ancestor is not None

            # Note that this handles the artificial root gracefully since it checks from the incomming outward in both directions. both the in and out from the artifical root (if root is the common ancestor) will get both increased or both decreased. (this becasue of the odd number of edges. All cycles *not* including root will instead be even number of edges)

            if degen:         
                # cant flip values because they aren't alternating (0, 1... ie. unused, used...), 
                # all we can do is insert the edge as value=0 (unused)

                # remove first zero edge on target side
                # WHY:
                # removed edge must be zero (tree must contain all used edges, duh)
                # removed edge must be on the target side, because incomming must attach through source, so that it is pointing away from the root (this the "strongly feasible" invariant which prevents non-termination)
                # it helps to remember that edges alternate direction, and fliping the edges (which is blocked in the degen case) also alternates. 
                # edges on target side are left alone, edges above cut are left alone, therefore the only viable cut is the first zero edge on the target side. 
                # counting from the first target side edge = 0, this must be an even edge, because the odd edges will have direction the same as the incomming and would (being currently attached on their target side) already be violating the invariant. 

                ii = t_i
                self.list.clear()
                self.list.append(ii)

                while self.value[ii] != 0:
                    if p:
                        print("blah", ii, self.parent[ii], common_ancestor, exiting_edge)
                    ii = self.parent[ii]
                    assert ii != common_ancestor, "should have found a zero"
                    assert ii != -1, "should have found common ancestor"
                    self.list.append(ii)
                assert ii == exiting_edge

                # ii is now the outgoing edge, we fix the graph going back toward the incomming
                N = len(self.list)
                for i in reversed(range(1, N)):
                    # a was closer to the incomming and is new parent
                    a = self.list[i - 1]
                    # b was closer to root and was the origina parent
                    b = self.list[i]

                    assert b != t_i

                    #originally parent[a] -> b, switch to parent[b] -> a
                    #starts with most distant, to not clobber
                    self.parent[b] = a
                    self.value[b] = self.value[a]
                    self.depth[b] = s_d + 1 + i
                    self.potential[b] += reduced_cost # whole fragment increases by same potential

                self.parent[t_i] = s_i
                self.value[t_i] = 0 # degen case, must insert as 0
                self.depth[t_i] = s_d + 1 # attached through source so depth is one greater
                self.potential[t_i] += reduced_cost
                assert self.potential[t_i] == self.potential[s_i] + cost # the potential invariant

                self.total_potential += reduced_cost * len(self.list)

            else:
                if p:
                    print("insert")
                # since we are flipping edges, any odd edge (pointing toward the root) must become zero, and so cannot stay pointing toward root. This means the odd edge closest to the root must be the one removed

                # furthermore, consider that the invariant means that the chain on the target side cannot be longer than 1, since this would mean there was (currently) a zero edge pointing toward root

                # now consider that the first target-side edge must be flipped to zero, and so must point away from the root, which it does. 
                # this restricts our edge removal to the source side.

                # ??? the invariant guarentees that the target side points to the root directly
                # this means the source side also connects to the root
                # based on whether root edge is a 1 or zero we must 

                assert common_ancestor == -1 or common_ancestor == t_i
                assert self.parent[t_i] == -1 or common_ancestor == t_i

                assert self.value[t_i] == 1 or common_ancestor == t_i
                self.value[t_i] = 0

                self.list.clear()
                self.list.append(s_i)
                ii = s_i
                while ii != common_ancestor:
                    ii = self.parent[ii]
                    self.list.append(ii)

                C = len(self.list)
                for i in reversed(range(1, C - 1)):
                    ii = self.list[i]
                    if i == C - 2:
                        assert self.parent[ii] == common_ancestor

                        # this connects to common ancestor, keep it if it becomes 1
                        if C % 2 == 1:
                            assert self.value[ii] == 0, "this should become 1"   
                            assert ii >= self.N
                            assert common_ancestor != t_i 
                            # Y<-0-R becomes Y<-1-R and must be kept. the X-1->Y which is next can be dropped as it becomes zero
                            self.value[ii] = 1
                            continue
                        else:
                            assert self.value[ii] == 1, "this should become 0"
                            assert ii < self.N 
                            # X-1->R becomes X-0->R and can be discarded. reconnect X through incomming edge
        
                    self.parent[ii] = self.list[i-1]

                    #self.value[ii] = # already is correct, coincidently
                    prev_val = self.value[self.list[i-1]] 
                    assert self.value[ii] == (1 if prev_val == 0 else 0), (s_i, t_i, i, ii, self.value[ii], prev_val)
                    assert self.value[ii] == (i - 1) % 2 

                    self.depth[ii] = t_d + 1 + i 
                    self.potential[ii] -= reduced_cost # same for whole subtree
        
                # this whole snippet gets reveresed and value flipped
                # the last edge connecting to common_ancestor is kept in place, the second to last is booted
                self.parent[s_i] = t_i
                self.depth[s_i] = t_d + 1
                self.value[s_i] = 1
                self.potential[s_i] -= reduced_cost
                assert self.potential[t_i] == self.potential[s_i] + cost # the potential invariant

                self.total_cost += int(reduced_cost)
                self.total_potential -= reduced_cost

        def reduced_cost(self, s_i, t_i):
            """the change in cost which would result from adding this edge"""

            # obviously cost increases by C(s,t), and decreases by C(s, p(s)) and C(t, p(t)) where p is current parent
            # and so on alternating up to the common ancestor
            # this is what potential precomputes
            # potential contribution of X->Y edge is always negative, Y->X always positive (where -> points toward root)
            # meaning that C(X,Y) + P(X) - P(Y) is the right answer for change in cost, as per logic above

            return self.cost(s_i, t_i) + self.potential[s_i] - self.potential[t_i]

        def cost(self, s_i, t_i):
            """square distance between rgb values"""
            if s_i == -1 or t_i == -1:
                return self.artificial_edge_cost

            assert s_i < self.N
            assert t_i >= self.N
            a = self.source[s_i]
            b = self.target[t_i - self.N]
            cost = np.sum( ( a - b ) ** 2)
            cost = int(cost * self.COST_SCALE)

            assert cost < self.artificial_edge_cost, (a, b, cost, self.artificial_edge_cost)
            return cost

        def solve_reference(self):
            """
            use scipy to solve the optimum transport, to verify correctness of our algo
            """

            from scipy.optimize import linear_sum_assignment
            cost = np.sum(
                (self.source[:, None, :] - self.target[None, :, :]) ** 2,
                axis=2
            )

            cost *= self.COST_SCALE
            cost = cost.astype(np.int32)

            rows, cols = linear_sum_assignment(cost)

            assignment = np.empty(self.N, dtype=np.int32)
            assignment[rows] = cols

            total_cost = cost[rows, cols].sum()

            return assignment, total_cost

        def check(self):        
            ref_map, ref_cost = self.solve_reference()

            ref_cost_sum = 0
            for s_i, t_i in enumerate(ref_map):
                ref_cost_sum += self.cost(s_i, t_i + self.N)

            assert ref_cost == ref_cost_sum
            assert self.total_cost == ref_cost
            print(self.map, ref_map)
            assert np.all(self.map == ref_map)


    return (NetworkSimplex,)


@app.cell
def _(NetworkSimplex):
    def rgb(image):
        """Convert an RGB image to float values on the [0, 1] cube."""
        image = np.asarray(image, dtype=float)
        return image / 255 if image.max() > 1 else image

    path = "./image.png"
    image = Image.open(path).convert("RGB")
    algo = NetworkSimplex(rgb(image)[:10,:10])
    return algo, image, rgb


@app.cell
def _(algo):
    algo.sort_initial_by_nearest()
    algo.faster_solve()
    algo.check()
    return


@app.cell
def _(NetworkSimplex, image, rgb):
    algo2 = NetworkSimplex(rgb(image))
    algo2.solve_reference()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
