from _bootstrap import bootstrap

bootstrap()

from video_super_resolution.config import ClipSpec
from video_super_resolution.cost import estimate_matrix

# Recommended 6-clip set, 10 s @ 25 fps = 250 frames each.
CLIPS = [
    ClipSpec("tears_of_steel_face", "face", 1280, 720, 25, 250),
    ClipSpec("big_buck_bunny_texture", "texture", 1280, 720, 25, 250),
    ClipSpec("sintel_motion", "fast-motion", 1280, 720, 25, 250),
    ClipSpec("pexels_neon_text", "text+lowlight", 1280, 720, 25, 250),
    ClipSpec("videolq_real", "real-degraded", 1280, 720, 25, 250, track="B"),
    ClipSpec("ai_generated_domain", "domain-match", 1280, 720, 25, 250),
]

TARGETS = {"1080p": (1920, 1080), "2K/1440p": (2560, 1440)}
BUDGET_USD = 20.0


def main() -> None:
    for label, (w, h) in TARGETS.items():
        rows, total = estimate_matrix(CLIPS, w, h, models=("topaz", "seedvr2"))
        print(f"\n=== target {label} ({w}x{h}) ===")
        for r in rows:
            print(f"  {r.clip:28s} {r.model:9s} ${r.usd:6.3f}")
        print(f"  {'TOTAL':28s} {'':9s} ${total:6.3f}   "
              f"({total / BUDGET_USD:5.1%} of ${BUDGET_USD:.0f}, "
              f"{'OK' if total <= BUDGET_USD else 'OVER BUDGET'})")


if __name__ == "__main__":
    main()
