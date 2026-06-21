#!/bin/bash
cd /Users/arvind/Developer/tinker-rl-lab/zvf-program

echo "Restoring original deck..."
python3 build_lightning_deck.py

echo "Applying Slide 1 enrichments..."
python3 enrich_title_slide.py
mv ZVF_Program_Progress_2026-06-14_lightning_enriched.pptx ZVF_Program_Progress_2026-06-14_lightning.pptx

echo "Applying Slide 2 enrichments..."
python3 enrich_slide_2.py
mv ZVF_Program_Progress_2026-06-14_lightning_enriched.pptx ZVF_Program_Progress_2026-06-14_lightning.pptx

echo "Applying Slide 3 enrichments..."
python3 enrich_slide_3.py

echo "Applying Slide 4 enrichments..."
python3 enrich_slide_4.py
mv ZVF_Program_Progress_2026-06-14_lightning_enriched.pptx ZVF_Program_Progress_2026-06-14_lightning.pptx

echo "Applying Slide 5 enrichments..."
python3 enrich_slide_5.py
mv ZVF_Program_Progress_2026-06-14_lightning_enriched.pptx ZVF_Program_Progress_2026-06-14_lightning.pptx

echo "Applying Slide 6 enrichments..."
python3 enrich_slide_6.py
mv ZVF_Program_Progress_2026-06-14_lightning_enriched.pptx ZVF_Program_Progress_2026-06-14_lightning.pptx

echo "Applying Slide 8 enrichments (and padding)..."
python3 enrich_slide_8.py

echo "Done! The fully enriched presentation is at ZVF_Program_Progress_2026-06-14_lightning_enriched.pptx"
