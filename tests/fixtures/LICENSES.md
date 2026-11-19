# Test fixture provenance

Real comic pages used to test panel detection. All three are labelled **Public domain** on
Wikimedia Commons (the licence is as stated there; this is not an independent legal opinion)
and originate from [Comic Book Plus](https://comicbookplus.com/), which hosts public-domain
comics. Files are the unmodified originals.

| File | Commons page | Comic Book Plus id | Why it is useful |
|---|---|---|---|
| `nature_boy_4.jpg` | https://commons.wikimedia.org/wiki/File:Pecos_Bill_in_Nature_Boy_4_page1.jpg | 18201 | aged paper, scan border, title banner, caption box overlapping a frame, narrow gutters |
| `western_fighters_6.jpg` | https://commons.wikimedia.org/wiki/File:Pecos_Bill_in_Western_Fighters_6_page1.jpg | 39853 | caption text sitting in the gutters between panels, title lettering |
| `western_fighters_8.jpg` | https://commons.wikimedia.org/wiki/File:Pecos_Bill_in_Western_Fighters_8_page1.jpg | 39966 | title banner, captions, circular inset panel |

The git-ignored `samples/` folder is for local, possibly copyrighted pages and is never committed.

## Pages added by the repository owner

`public_001.jpg` to `public_005.jpg` were added by the repository owner, who obtained them from
[Comic Book Plus](https://comicbookplus.com/), a site that hosts comics it treats as public domain,
and stated they are in the public domain. The files carry no EXIF data; their JPEG comment
(`gd-jpeg`) and 975 px width are consistent with that source.

**The specific title, issue and page URL of each file was not recorded.** Add them here (and the
basis for its public-domain status) so the provenance can be checked per page.

| File | What it is | Why it is useful |
|---|---|---|
| `public_001.jpg` | black-and-white newsprint public-service page | no colour, thin frames, title banner, 2-3-2 grid |
| `public_002.jpg` | colour action page | L-shaped panel wrapping an inset, irregular layout |
| `public_003.jpg` | colour action page, dense art | slanted gutters, captions on frame lines |
| `public_004.jpg` | colour page on aged paper | 3x2 grid, floating captions, partly unframed art |
| `public_005.jpg` | colour page on aged paper with a scan edge | 3x2 grid, floating captions |

Each page has a hand-made ground-truth file next to it (`<name>.labels.json`, format in
`docs/evaluation.md`) used by the evaluation harness.
