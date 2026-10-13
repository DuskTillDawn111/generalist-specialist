# Module14B final OCR discovery source audit

- [PARSeq_STRHub](https://github.com/baudm/parseq/tree/1902db043c029a7e03a3818c616c06600af574be)
- [Release](https://github.com/baudm/parseq/releases/tag/v1.0.0)
- [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177.pdf)
- [SVTRv2](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md)
- [MATRN](https://github.com/byeonghu-na/MATRN)
- [VisionLAN](https://github.com/wangyuxin87/VisionLAN)
- [ABINet](https://github.com/FangShancheng/ABINet)

Only primary publisher/paper evidence. Complete11 candidate metadata and four-slot freeze are in ocr_gapfill_candidate_universe.csv / ocr_gapfill_shortlist.json. Exact reproduced release-specific quality is not asserted where not published. Public IIIT5K3000 scores never replace local frozen200 results. Native publisher source commit and v1.0.0 release filenames/hash prefixes were fixed before any new GPU run.

Task remains supplied-word recognition only, native pretrained recognizer RGB preprocessing, no detection, crop generation, lexicon, dataset filtering, retraining, manual slowdown, memory scaling or decoder tuning. The last authorized OCR search round is exhausted by this frozen list; no third search round or replacement after outcomes.
