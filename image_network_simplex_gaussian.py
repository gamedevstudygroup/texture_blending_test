import marimo

__generated_with = "0.23.16"
app = marimo.App(css_file="./custom.css")


@app.cell
def _():
    from pathlib import Path
    from time import perf_counter

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from PIL import Image

    import texture_gaussianization as tg

    # These imports are kept inside the notebook so the file can be opened as a
    # standalone marimo app without requiring a separate support module.
    return Image, Path, go, mo, np, perf_counter, tg


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Image transport to a Gaussian distribution

    This notebook implements the gaussianization transform from https://inria.hal.science/hal-01824773/document

    The image/histogram preview are bullshit ai code. The algorithm I wrote by hand to understand better.
    The end goal is to write a parallel version for the GPU, or perhaps multithreaded workers on cpu, idk.

    Network simplex is a weird algorithm using a "graph" to make irreversible progress toward an optimum mapping of two sets, where each potential element mapping (here between some pixel in the texture image and some pixel in the target gaussian image) has a certain cost. It has some bullshit terms like basis spanning tree (I don't see how its a basis) and flow (which in our case is always at all stages either 0 or 1) and strongly feasible (which I've made no sense of the the name, it is just an invariant which helps prevent non-termination)
    """)
    return


@app.cell
def _(Path, mo):
    image_dir = Path("images")

    files = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file()
    )
    file_options = {str(p.relative_to(image_dir)): p for p in files}

    dropdown = mo.ui.dropdown(
        options=file_options,
        label="Image",
        value=list(file_options.keys())[3]
    )

    dropdown
    return (dropdown,)


@app.cell
def _(Image, Path, dropdown, np):
    image_path = dropdown.value
    path = Path(image_path).expanduser()

    if not path.exists():
        raise OSError(f"no file {path}")

    # Converting to RGB gives a predictable final shape: (height, width, 3).
    image = Image.open(path).convert("RGB")
    image_array = np.asarray(image)
    return image, image_array


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## image high pass filter
    textures must be stationary to be used with out method

    The best I have come up with is a repeated gaussian. there are other techniques

    TODO: apply histogram correction after gaussian in order to restore high freq contrast.
    """)
    return


@app.cell
def _(mo):
    low_freq_sigma = mo.ui.number(value=16)
    return (low_freq_sigma,)


@app.cell
def gaussian_high_pass_helpers(
    channel_histograms,
    go,
    image,
    low_freq_sigma,
    mo,
    np,
):
    from scipy.ndimage import gaussian_filter

    def low_pass(data):
        """Blur RGB data spatially without mixing its color channels."""
        sigma = low_freq_sigma.value
        return gaussian_filter(data, sigma=(sigma, sigma, 0))

    def subtract_low_frequencies(data, low = None):
        """Remove one Gaussian low-frequency estimate while preserving the mean."""
        low = low_pass(data) if low is None else low
        return data - low + low.mean(axis=(0, 1), keepdims=True)

    def show_filtering(*, controls, columns):
        """Render labeled filtering results with a common clipped final preview."""
        bins = 256

        def small_histogram(data, dots = False):
            centers = (np.arange(bins) + 0.5) / bins
            fig = go.Figure()

            from scipy.ndimage import gaussian_filter1d

            # Histogram and preview use the same final display conversion.
            histogram = channel_histograms(np.clip(data, 0, 255), bins)
            if not dots:
                histogram = gaussian_filter1d(histogram, bins / 128, axis=1)


            for channel, color in enumerate(("red", "green", "blue")):
                if dots:
                    mask = histogram[channel] != 0
                    x = centers[mask]
                    y = histogram[channel][mask]
                else:
                    x = centers
                    y = histogram[channel]
        
                fig.add_scatter(
                    x=x,
                    y=y,
                    mode="markers" if dots else "lines",
                    line_color=color,
                    showlegend=False,
                )

            fig.update_layout(
                xaxis={"range": [0, 1]},
                #yaxis={"range": [0, ymax]},
                margin={"l": 20, "r": 5, "b": 20, "t": 5},
                height=180,
            )

            return mo.ui.plotly(
                fig,
                config={
                    "staticPlot": True,
                    "displayModeBar": False,
                },
            )

        def result_column(title, result, remaining):
            # Do not alter filtering data here: clipping is solely for output.
            preview = np.clip(result, 0, 255).astype(np.uint8)
            #remaining_preview = np.clip(remaining, 0, 255).astype(np.uint8)
            # based on how marimo shows f64 images
            lo = remaining.min()
            hi = remaining.max()
            r_scale = 255.0 / (hi - lo)
            return mo.vstack([
                mo.md(f"**{title}**"),
                mo.image(preview),
                small_histogram(result),
                mo.md(f"**Low frequency remaining scale={r_scale:f}**"),
                mo.image(remaining),
                small_histogram(remaining, dots=True),
            ])

        return mo.vstack(
            [
                mo.hstack([image.size, *controls], justify="start", gap=1),
                mo.hstack(
                    [
                        result_column(title, result, remaining)
                        for title, result, remaining in columns
                    ]
                ),
            ]
        )

    return low_pass, show_filtering, subtract_low_frequencies


@app.cell
def gaussian_high_pass(
    image_array,
    low_freq_sigma,
    low_pass,
    np,
    show_filtering,
    subtract_low_frequencies,
):
    """Compare one standard high-pass pass with its fitted-gain variant."""
    image_f = image_array.astype(np.float64)
    low = low_pass(image_f)
    subtract = subtract_low_frequencies(image_f, low)

    image_centered = image_f - image_f.mean(axis=(0, 1), keepdims=True)
    low_centered = low - low.mean(axis=(0, 1), keepdims=True)

    # Maximum low-frequency removal gain before the residual becomes negatively 
    # correlated with the estimated low-frequency component.
    gain = np.divide(
        np.mean(image_centered * low_centered, axis=(0, 1), keepdims=True),
        np.mean(low_centered**2, axis=(0, 1), keepdims=True),
        out=np.zeros_like(low_centered.mean(axis=(0, 1), keepdims=True)),
        where=np.mean(low_centered**2, axis=(0, 1), keepdims=True) > 0,
    )
    subtract_gain = image_f - gain * low_centered

    show_filtering(
        controls=[low_freq_sigma],
        columns=[
            ("Original", image_f, low),
            ("Subtract low frequencies", subtract, low_pass(subtract)),
            (
                "Subtract low frequencies with gain",
                subtract_gain,
                low_pass(subtract_gain),
            ),
        ],
    )
    return (image_f,)


@app.cell
def _(mo):
    # Exposed so repeated filtering can be tuned without changing notebook code.
    high_pass_cycles = mo.ui.number(value=2, start=1, step=1, label="High-pass cycles")
    return (high_pass_cycles,)


@app.cell
def gaussian_high_pass_repeated(
    high_pass_cycles,
    image_f,
    low_freq_sigma,
    low_pass,
    show_filtering,
    subtract_low_frequencies,
):
    """Repeat subtractive Gaussian high-pass filtering without intermediate clips."""
    cycles = max(1, int(high_pass_cycles.value or 1))
    repeated = image_f
    for _ in range(cycles):
        # Keep full precision between cycles; show_filtering clips only at output.
        repeated = subtract_low_frequencies(repeated)

    repeated_high_pass_view = show_filtering(
        controls=[low_freq_sigma, high_pass_cycles],
        columns=[
            ("Original", image_f, low_pass(image_f)),
            (
                f"Repeated Gaussian high-pass ({cycles} cycles)",
                repeated,
                low_pass(repeated),
            ),
        ],
    )
    repeated_high_pass_view  # noqa: B018 - Marimo renders the cell's final expression.
    return


@app.cell
def _(go, image_array, mo, np):
    # codex generated ui
    height, width, _ = image_array.shape
    # A sparse transparent scatter layer activates Plotly's rectangular
    # selection tool without making the image preview expensive to render.
    stride = max(1, int(np.ceil(np.sqrt(height * width / 20_000))))
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    image_figure = go.Figure(go.Image(z=image_array))
    image_figure.add_trace(go.Scattergl(
        x=columns.ravel(),
        y=rows.ravel(),
        mode="markers",
        marker={"size": 3, "opacity": 0.001},
        hoverinfo="skip",
        showlegend=False,
    ))
    image_figure.update_layout(
        dragmode="select",
        height=520,
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        newselection={"line": {"color": "#f5c542", "width": 2}},
        activeselection={"fillcolor": "rgba(245, 197, 66, 0.18)"},
        selectionrevision="image-roi",
    )
    image_figure.update_xaxes(range=(-0.5, width - 0.5), visible=False)
    image_figure.update_yaxes(
        range=(height - 0.5, -0.5),
        scaleanchor="x",
        visible=False,
    )
    image_selector = mo.ui.plotly(image_figure)

    # has to be defined outside of the cell it is used in.
    histo_bins = mo.ui.number(step=1, value=256)
    return histo_bins, image_selector


@app.cell
def _(dropdown, go, histo_bins, image_array, image_selector, mo, np):
    # codex generated ui and histogram
    def rgb(image):
        """Convert an RGB image to float values on the [0, 1] cube."""
        image = np.asarray(image, dtype=float)
        return image / 255 if image.max() > 1 else image

    def channel_histograms(image, bins):
        """Return one normalized histogram per RGB channel."""
        return np.array([
            np.histogram(values, bins=bins, range=(0, 1))[0] / values.size
            for values in rgb(image).reshape(-1, 3).T
        ])

    def sweep_histogram_bounds(
        reference_image, window_shape, *, bins, sigma=2, max_windows=400
    ):
        """Sample windows, smooth their histograms, then return per-bin bounds."""
        window_height, window_width = window_shape
        max_y = reference_image.shape[0] - window_height
        max_x = reference_image.shape[1] - window_width

        aspect = (max_x + 1) / max(max_y + 1, 1)
        y_count = max(1, int(np.sqrt(max_windows / max(aspect, 1e-6))))
        x_count = max(1, max_windows // y_count)

        y_positions = np.unique(np.linspace(0, max_y, y_count, dtype=int))
        x_positions = np.unique(np.linspace(0, max_x, x_count, dtype=int))

        samples = np.array([
            channel_histograms(
                reference_image[y:y + window_height, x:x + window_width], bins
            )
            for y in y_positions
            for x in x_positions
        ])

        from scipy.ndimage import gaussian_filter1d
        samples = gaussian_filter1d(samples, sigma, axis=2)

        return samples.min(axis=0), samples.max(axis=0)

    def view_histogram(image, *, bins=256, sweep_source=None, sigma=2):
        selected = channel_histograms(image, bins)
        centers = (np.arange(bins) + 0.5) / bins

        lower = upper = None
        if sweep_source is not None:
            shape = image.shape[:2]
            if shape == sweep_source.shape[:2]:
                shape = (shape[0] // 2, shape[1] // 2)

            lower, upper = sweep_histogram_bounds(sweep_source, shape, bins=bins)
            # smooth the histograms instead
            # lower = gaussian_filter1d(lower, sigma, axis=1)
            # upper = gaussian_filter1d(upper, sigma, axis=1)

        figure = go.Figure()

        for channel, (color, fill, name) in enumerate((
            ("red",   "rgba(255,0,0,.14)",   "Red"),
            ("green", "rgba(0,255,0,.14)",   "Green"),
            ("blue",  "rgba(0,0,255,.14)",   "Blue"),
        )):
            if lower is not None:
                figure.add_scatter(
                    x=centers, y=lower[channel],
                    mode="lines", line_color=color,
                    legendgroup="sweep", showlegend=False,
                )
                figure.add_scatter(
                    x=centers, y=upper[channel],
                    mode="lines", line_color=color,
                    fill="tonexty", fillcolor=fill,
                    legendgroup="sweep",
                    name="Sweep envelope",
                    showlegend=channel == 0,
                )

            figure.add_scatter(
                x=centers, y=selected[channel],
                mode="markers", name=name,
                marker={"color": color, "size": 5},
            )

        figure.update_layout(
            legend=dict(
                groupclick= "togglegroup",
                orientation="h",      # Makes the legend horizontal
                yanchor="bottom",     # Anchors the legend by its bottom edge
                y=-0.2,               # Positions it below the x-axis (adjust as needed)
                xanchor="center",     # Anchors the legend by its center
                x=0.5                 # Centers it horizontally
            ),
            xaxis={"title": "Intensity", "range": (0, 1)},
            yaxis_title="Probability mass",
            margin={"l": 0, "r": 0, "b": 0, "t": 30},
        )

        return figure

    def view_histogram_3d(image, *, bins=16):
        """Plot occupied RGB histogram bins as interactive, frequency-sized dots."""
        histogram, edges = np.histogramdd(rgb(image).reshape(-1, 3), bins=bins, range=((0, 1),) * 3)
        centers = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
        grid = np.meshgrid(*centers, indexing="ij")
        occupied = histogram > 0
        points = np.column_stack([axis[occupied] for axis in grid])
        counts = histogram[occupied]

        # Bin-center RGB colors make the geometry and color distribution agree.
        colors = [f"rgb({r:.0f}, {g:.0f}, {b:.0f})" for r, g, b in points * 255]
        figure = go.Figure(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            marker={"color": colors, "size": 3 + 17 * np.sqrt(counts / counts.max())},
            text=counts, hovertemplate="R: %{x:.3f}<br>G: %{y:.3f}<br>B: %{z:.3f}<br>Pixels: %{text}<extra></extra>",
        ))
        figure.update_layout(
            dragmode="turntable",
            scene={"xaxis_title": "Red", "yaxis_title": "Green", "zaxis_title": "Blue", "aspectmode": "cube"},
            margin={"l": 0, "r": 0, "b": 0, "t": 0},
        )
        return figure

    selection_range = image_selector.ranges
    x_range = selection_range.get("x")
    y_range = selection_range.get("y")
    if x_range is not None and y_range is not None and len(x_range) == len(y_range) == 2:
        x0, x1 = np.clip(np.floor(sorted(x_range)).astype(int), 0, image_array.shape[1] - 1)
        y0, y1 = np.clip(np.floor(sorted(y_range)).astype(int), 0, image_array.shape[0] - 1)
        selected_pixels = image_array[y0:y1 + 1, x0:x1 + 1]
    else:
        selected_pixels = image_array

    histogram_figure = view_histogram(
        selected_pixels, bins=histo_bins.value, sigma=2.0 * histo_bins.value / 256, sweep_source=image_array
    )
    histogram_3d_figure = view_histogram_3d(selected_pixels)

    mo.vstack(
        [
            mo.hstack([dropdown, histo_bins]),
            mo.hstack([
                image_selector,
                mo.ui.tabs({"2D channels": histogram_figure, "3D RGB bins": histogram_3d_figure})],
                widths=[1,3],
                justify="start",
                align="start",
                gap=1,
            ),
        ],
    )
    return channel_histograms, view_histogram


@app.cell
def _(Image, histo_bins, image, mo, np, tg, view_histogram):
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

    # UNUSED
    def gen_gaussian2(size, mean=0.5, std=1/6, seed=None):
        """DOES NOT WORK Generate image with gaussian distribution directly. """
        h, w = size 
        a = np.arange(w * h).reshape(h*w, 1)
        a = np.repeat(a, 3, axis=1)

        # task, fill pixel values with uniform 3d grid values
        m = int(np.floor(np.cbrt(h*w)))
        M = m ** 3

        i = np.arange(M)
        a = i % m 
        b = (i // m) % m
        c = i // (m*m)

        # Put samples at centers of grid cells.
        u = np.stack((a,b,c), axis=1)
        rng = np.random.default_rng(seed) # without this does not work
        u = (u + rng.random((M,3))) / m

        # Uniform [0,1] -> Gaussian.
        from scipy.special import erfinv
        g = mean + std * np.sqrt(2) * erfinv(2 * u - 1)

        out = np.zeros((h*w, 3))
        out[:M] = g
        return out.reshape(h,w,3)

    # gaussian_image = gen_gaussian(image.size)
    gaussian_image = tg.gaussian_reference(image.size) #this method is a bit better, but returns float64 and can be outside [0,1)

    gaussian_preview = Image.fromarray(
        np.rint(gaussian_image * 256).astype(np.uint8), mode="RGB"
    )

    # Reuse the same Plotly histogram styling without the source-image sweep.
    gaussian_histogram = view_histogram(gaussian_preview, bins=histo_bins.value)

    mo.hstack(
        [mo.image(gaussian_preview, width=300), gaussian_histogram],
        justify="start",
        align="start",
        gap=1,
        widths=[1, 3],
    )
    return


@app.cell
def _(mo):
    # Expose the visual/metric tuning knobs without coupling the fast separable
    # transform to a reduced-resolution copy of the source texture.
    mapping_seed = mo.ui.number(
        start=0, stop=1_000_000, step=1, value=1, label="Accuracy reference seed"
    )
    accuracy_bins = mo.ui.slider(
        steps=[4, 6, 8, 10, 12, 16],
        value=8,
        show_value=True,
        label="3D quantile bins",
    )
    use_pca = mo.ui.checkbox(value=True, label="PCA decorrelated colorspace")
    use_compression_correction = mo.ui.checkbox(
        value=False, label="DXT compression correction"
    )

    mo.vstack(
        [
            mo.md("## GPU Zen 2 Gaussianization controls"),
            mo.hstack(
                [mapping_seed, accuracy_bins],
                justify="start",
                gap=1,
            ),
            mo.hstack(
                [use_pca, use_compression_correction],
                justify="start",
                gap=1,
            ),
        ]
    )
    return accuracy_bins, mapping_seed, use_compression_correction, use_pca


@app.cell
def _(
    Image,
    accuracy_bins,
    image_array,
    mapping_seed,
    mo,
    np,
    perf_counter,
    tg,
    use_compression_correction,
    use_pca,
):
    # The chapter's separable method is O(N log N), so it runs on every source
    # pixel. The independent 3D sample is used only by the retained accuracy
    # metric; it is not an input to the rank mapping itself.
    gaussian_target = tg.gaussian_reference(
        image_array.shape[0] * image_array.shape[1],
        seed=int(mapping_seed.value),
    )
    gpu_started = perf_counter()
    gpu_result = tg.separable_gaussianize(
        image_array,
        reference=gaussian_target,
        decorrelate=use_pca.value,
        compression_correction=use_compression_correction.value,
    )
    gpu_seconds = perf_counter() - gpu_started
    # Compression correction is a reversible storage transform. Accuracy is
    # therefore measured on canonical values after undoing that scaling.
    gpu_metrics = tg.distribution_metrics(
        gpu_result.undo_compression_correction(),
        gaussian_target,
        bins=int(accuracy_bins.value),
    )
    gpu_preview = Image.fromarray(
        np.rint(np.clip(gpu_result.texture, 0.0, 1.0) * 255).astype(np.uint8),
        mode="RGB",
    )
    mo.vstack(
        [
            mo.md(
                "## GPU Zen 2 separable mapping\n"
                f"Mapped all **{image_array.shape[0] * image_array.shape[1]:,}** "
                f"pixels in **{gpu_seconds:.4f} s**. PCA is "
                f"**{'on' if use_pca.value else 'off'}** and compression "
                f"correction is **{'on' if use_compression_correction.value else 'off'}**. "
                f"Storage scales: `{np.array2string(gpu_result.compression_scale, precision=4)}`.\n\n"
                f"3D JS error: **{gpu_metrics.reference_js:.6f}**; excess "
                f"joint dependence: **{gpu_metrics.excess_dependence_js:.6f}**."
            ),
            mo.image(gpu_preview, width=420),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
