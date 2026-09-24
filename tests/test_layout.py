import unittest

from PIL import Image, ImageDraw

from layout import (build_lines, coverage, crop_box, ink_stats, residual_regions,
                    segment_boxes, tight_box)


def detection(box):
    a, b, c, d = box
    return {"polygon": [[a, b], [c, b], [c, d], [a, d]], "score": .9}


def page(rows, size=(240, 160)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1 in rows:
        for x in range(x0, x1 - 3, 6):
            draw.rectangle((x, y0, x + 2, y1 - 1), fill="black")
    return image


class LayoutTests(unittest.TestCase):
    def test_clamp_and_reject_empty(self):
        self.assertEqual(crop_box([-3, 2.2, 250, 170], (240, 160)), [0, 2, 240, 160])
        for box in ([250, 0, 260, 20], [0, 0, float("nan"), 20], [0, 2, 2, 2]):
            with self.assertRaises(ValueError):
                crop_box(box, (240, 160))

    def test_split_tall_and_suppress_fragments_after_split(self):
        image = page([(10, 10, 200, 18), (10, 30, 200, 38)])
        lines, suppressed = build_lines(image, [detection([10, 8, 200, 40]),
            detection([10, 9, 80, 19]), detection([10, 29, 80, 39])], {})
        self.assertEqual(len(lines), 2)
        self.assertEqual([x["box"][2] for x in lines], [200, 200])
        self.assertEqual(len(suppressed), 2)

    def test_same_row_partial_overlap_retains_extra_text(self):
        image = page([(10, 20, 160, 28)])
        lines, _ = build_lines(image, [detection([10, 19, 90, 29]), detection([80, 19, 160, 29])], {})
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["box"][::2], [10, 160])

    def test_large_partial_overlap_preserves_both_ends(self):
        image = page([(10, 20, 160, 28)])
        lines, _ = build_lines(image, [detection([10, 19, 120, 29]), detection([60, 19, 160, 29])], {})
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["box"][::2], [10, 160])

    def test_small_detached_dots_do_not_split_a_single_row(self):
        image = page([(10, 10, 200, 18)])
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 5, 22, 6), fill="black")
        draw.rectangle((50, 5, 52, 6), fill="black")
        lines, _ = build_lines(image, [detection([10, 3, 200, 20])], {})
        self.assertEqual(len(lines), 1)
        self.assertLessEqual(lines[0]["box"][1], 5)

    def test_columns_stay_separate_and_read_top_to_bottom(self):
        boxes = [[10, 10, 100, 20], [105, 10, 200, 20],
                 [10, 35, 100, 45], [105, 35, 200, 45]]
        lines, _ = build_lines(page(boxes), [detection(b) for b in boxes], {})
        self.assertEqual(len(lines), 4)
        self.assertEqual([r["box"][0] for r in lines], [10, 10, 105, 105])
        self.assertEqual([r["order"] for r in lines], list(range(4)))
        self.assertEqual(len({r["column"] for r in lines}), 2)

    def test_spanning_heading_precedes_columns(self):
        boxes = [[10, 3, 205, 11], [10, 20, 90, 30], [125, 20, 205, 30],
                 [10, 45, 90, 55], [125, 45, 205, 55]]
        lines, _ = build_lines(page(boxes), [detection(b) for b in boxes], {})
        self.assertEqual([r["box"][0] for r in lines], [10, 10, 10, 125, 125])
        self.assertEqual(lines[0]["box"][2], 205)

    def test_residual_half_line_roi_includes_existing_fragment(self):
        image = page([(10, 20, 210, 28), (10, 45, 210, 53)])
        lines = [{"box": [110, 19, 210, 29]}, {"box": [110, 44, 210, 54]}]
        before = coverage(image, lines)
        self.assertGreater(before["uncovered_ink_fraction"], .3)
        rois = residual_regions(image, lines, {})
        self.assertEqual(len(rois), 1)
        self.assertLessEqual(rois[0][0], 10)
        self.assertGreaterEqual(rois[0][2], 210)
        self.assertLessEqual(rois[0][1], 20)
        self.assertGreaterEqual(rois[0][3], 53)

    def test_residual_rois_do_not_cross_existing_column_gutter(self):
        boxes = [[10, 20, 90, 30], [135, 20, 220, 30],
                 [10, 50, 90, 60], [135, 50, 220, 60]]
        image = page(boxes + [(0, 20, 10, 30), (220, 50, 235, 60)])
        lines = [{"box": b} for b in boxes]
        rois = residual_regions(image, lines, {})
        self.assertTrue(rois)
        self.assertTrue(all(b[2] <= 113 or b[0] >= 112 for b in rois))

    def test_color_noise_and_solid_shapes_do_not_create_rows(self):
        image = Image.new("RGB", (240, 160), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((5, 5, 230, 15), fill="red")
        draw.rectangle((5, 30, 230, 45), fill="black")
        self.assertEqual(residual_regions(image, [], {}), [])
        self.assertGreater(coverage(image, [])["colored_noise_fraction"], 0)

    def test_residual_budget_is_finite(self):
        image = page([(10, y, 180, y + 5) for y in (10, 60, 110)])
        self.assertEqual(len(residual_regions(image, [], {"max_residual_regions": 2})), 2)
        self.assertEqual(residual_regions(image, [], {"max_residual_regions": 0}), [])

    def test_tight_crop_preserves_top_bottom_and_x_coverage(self):
        image = page([(10, 20, 200, 28)])
        self.assertEqual(tight_box(image, [5, 10, 210, 40]), [5, 19, 210, 29])
        self.assertEqual(tight_box(Image.new("RGB", (30, 20), "white"), [0, 0, 30, 20]), [0, 0, 30, 20])

    def test_segments_cover_line_and_overlap_without_crossing_bounds(self):
        image = page([(10, 20, 430, 29)], size=(440, 60))
        boxes = segment_boxes(image, [10, 19, 430, 30])
        self.assertEqual(len(boxes), 3)
        self.assertEqual(boxes[0][0], 10)
        self.assertEqual(boxes[-1][2], 430)
        for a, b in zip(boxes, boxes[1:]):
            self.assertGreater(a[2], b[0])
        for box in boxes:
            self.assertTrue(10 <= box[0] < box[2] <= 430)
            self.assertEqual(box[1::2], [19, 30])
        self.assertEqual(segment_boxes(image, [10, 19, 430, 30], max_segments=1), [[10, 19, 430, 30]])

    def test_segments_prefer_a_nearby_whitespace_cut(self):
        image = page([(0, 10, 140, 20), (160, 10, 300, 20)], size=(300, 30))
        boxes = segment_boxes(image, [0, 9, 300, 21], max_segments=2, overlap=0)
        self.assertTrue(140 <= boxes[0][2] <= 160)
        self.assertEqual(boxes[0][2], boxes[1][0])

    def test_ink_stats_distinguish_long_text_from_single_letter(self):
        long_line = page([(3, 4, 225, 12)], size=(240, 20))
        letter = page([(3, 4, 8, 12)], size=(20, 20))
        self.assertTrue(ink_stats(long_line)["text_like"])
        self.assertGreater(ink_stats(long_line)["extent_ratio"], 12)
        self.assertLess(ink_stats(letter)["extent_ratio"], 2)


if __name__ == "__main__":
    unittest.main()
