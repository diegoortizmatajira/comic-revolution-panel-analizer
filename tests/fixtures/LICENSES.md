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

Each page has a hand-made ground-truth file next to it (`<name>.labels.json`, format in
`docs/evaluation.md`) used by the evaluation harness.
