# Bibliography: Histogram-Preserving Texture Tiling and Optimal Transport

Chronological order. Each entry has a project/publisher landing page and a direct paper link where available.

## Citation graph

Solid arrow = the citing paper shares at least one author with the cited paper.  
Dotted arrow = citation with no shared author.

```mermaid
flowchart TD
    B11["Bonneel et al. 2011<br/>Displacement Interpolation"]
    H18["Heitz & Neyret 2018<br/>High-Performance By-Example Noise"]
    D19["Deliot & Heitz 2019<br/>Procedural Stochastic Textures"]
    B19["Burley 2019<br/>Histogram-Preserving Blending"]
    G21["Guthe & Thuerck 2021<br/>Algorithm 1015"]
    S23["Bonneel & Digne 2023<br/>OT Survey"]

    D19 --> H18
    B19 -.-> H18
    B19 -.-> D19
    S23 --> B11
    S23 -.-> H18

    click B11 "#bonneel-et-al-2011" "Jump to Bonneel et al. 2011"
    click H18 "#heitz-neyret-2018" "Jump to Heitz & Neyret 2018"
    click D19 "#deliot-heitz-2019" "Jump to Deliot & Heitz 2019"
    click B19 "#burley-2019" "Jump to Burley 2019"
    click G21 "#guthe-thuerck-2021" "Jump to Guthe & Thuerck 2021"
    click S23 "#bonneel-digne-2023" "Jump to Bonneel & Digne 2023"
```

The graph only shows citation relationships between papers in this bibliography that I could verify. Nodes without arrows are still included because they are relevant papers discussed alongside the texture/OT work.

<a id="bonneel-et-al-2011"></a>
## 2011 — Nicolas Bonneel, Michiel van de Panne, Sylvain Paris, Wolfgang Heidrich

**Displacement Interpolation Using Lagrangian Mass Transport**

- [Landing / project page](https://www.cs.ubc.ca/labs/imager/tr/2011/DisplacementInterpolation/)
- [Paper (PDF)](https://www.cs.ubc.ca/labs/imager/tr/2011/DisplacementInterpolation/DisplacementSigAsia.pdf)
- [ACM / DOI landing](https://doi.org/10.1145/2024156.2024192)

<a id="heitz-neyret-2018"></a>
## 2018 — Eric Heitz, Fabrice Neyret

**High-Performance By-Example Noise using a Histogram-Preserving Blending Operator**

- [Landing / project page](https://eheitzresearch.wordpress.com/722-2/)
- [Paper (PDF)](https://hal.inria.fr/hal-01824773/document)
- [DOI](https://doi.org/10.1145/3233304)

<a id="deliot-heitz-2019"></a>
## 2019 — Thomas Deliot, Eric Heitz

**Procedural Stochastic Textures by Tiling and Blending**

- [Landing / project page](https://eheitzresearch.wordpress.com/738-2/)
- [Paper / GPU Zen 2 chapter (PDF)](https://drive.google.com/file/d/1QecekuuyWgw68HU9tg6ENfrCTCVIjm6l/view?usp=drive_open)

<a id="burley-2019"></a>
## 2019 — Brent Burley

**On Histogram-Preserving Blending for Randomized Texture Tiling**

- [Landing / JCGT article page](https://www.jcgt.org/published/0008/04/02/)
- [Paper — low-resolution PDF](https://www.jcgt.org/published/0008/04/02/paper-lowres.pdf)
- [Paper — full-resolution PDF](https://www.jcgt.org/published/0008/04/02/paper.pdf)

<a id="guthe-thuerck-2021"></a>
## 2021 — Stefan Guthe, Daniel Thuerck

**Algorithm 1015: A Fast Scalable Solver for the Dense Linear (Sum) Assignment Problem**

- [Landing / Fraunhofer publication page](https://publica.fraunhofer.de/entities/publication/5b53c32b-f328-4462-ae51-0368201bef71)
- [ACM / DOI landing](https://doi.org/10.1145/3442348)
- [Paper (PDF)](https://www.culip.org/assets/files/2021_guthe_lap.pdf)
- [Official CALGO implementation archive / index](https://calgo.acm.org/)

<a id="bonneel-digne-2023"></a>
## 2023 — Nicolas Bonneel, Julie Digne

**A Survey of Optimal Transport for Computer Graphics and Computer Vision**

- [Landing / Eurographics Digital Library](https://diglib.eg.org/items/65f57f22-a70f-49ac-9337-06ef9b70348b)
- [Paper (HTML)](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14778)
- [Paper (PDF)](https://diglib.eg.org/bitstream/handle/10.1111/cgf14778/v42i2pp439-460_cgf14778.pdf)
