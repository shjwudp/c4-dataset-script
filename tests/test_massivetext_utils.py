import unittest

from pyspark.sql import SparkSession
from c4_dataset_script.massivetext_utils import docs_dedup, is_repetition_removal


class SparkTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession
                     .builder
                     .master("local[*]")
                     .appName("Unit-tests")
                     .getOrCreate())

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_docs_dedup(self):
        input_data = [
            {
                "id": "url_1",
                "text": "Multilingual C4 (mC4) has 101 languages and is generated from 71 Common Crawl dumps.",
            },
            {
                "id": "url_2",
                "text": "C4 is a colossal, cleaned version of Common Crawl's web crawl corpus."
            },
            {
                "id": "url_3",
                "text": "C4 is a colossal, cleaned version of Common Crawl's web crawl dataset."
            },
        ]
        input_df = self.spark.createDataFrame(data=input_data)
        expected_deduplicated_items_set = [
            self.spark.createDataFrame(
                [{"id": "url_1"}, {"id": "url_2"}]).collect(),
            self.spark.createDataFrame(
                [{"id": "url_1"}, {"id": "url_3"}]).collect(),
        ]
        expected_duplicate_pairs_set = [
            self.spark.createDataFrame(
                [{"A_id": "url_2", "B_id": "url_3"}]).collect(),
            self.spark.createDataFrame(
                [{"A_id": "url_3", "B_id": "url_2"}]).collect(),
        ]

        self.spark.createDataFrame(
            [{"A_id": "url_2"}, {"B_id": "url_3"}]).collect()

        deduplicated_items, duplicate_pairs = docs_dedup(input_df, ngram=3)

        self.assertIn(sorted(deduplicated_items.collect()),
                      expected_deduplicated_items_set)
        self.assertIn(duplicate_pairs.collect(), expected_duplicate_pairs_set)

    def test_is_repetition_removal(self):
        junk_list = [
            "Image Title: Pottery Barn Counter Stools In Decker Leather Seat Barstool Inside Bar Plans 8. Filename: pottery-barn-counter-stools-in-decker-leather-seat-barstool-inside-bar-plans-8.jpg. Image Dimension: 710 x 639 pixels. Images Format: jpg/jpeg. Publisher/Author: Zackary Nikolaus. Uploaded Date: Monday - May 14th. 2018 18:35:16 PM. Category: Architecture. Image Source: interior.com.\nTap The Thumbnail Bellow to See Related Gallery of \"Pottery Barn Counter Stools In Decker Leather Seat Barstool Inside Bar Plans 8\"",
            "Embrace world class facilities at East Bourne Resort & Spa Shimla. Facilities at East Bourne Resort & Spa Shimla comprise multi cuisine restaurant, tours and travel desk. Avail facilities of East Bourne Resort & Spa Shimla.",
            "Using Opera 43.2442.1144 (PGO). Any ideas on when I might be able to export my bookmarks? Still unable to do so. I read here that the feature was added to Opera but my 'About Opera' keeps telling me I have the latest update. Thanks.\nAny ideas on when I might be able to export my bookmarks?\nNope. You can show your support for it in Suggestions Box.",
        ]
        ok_list = [
            "Biomedics 1 Day Extra are daily replacement disposable contact lenses by CooperVision Hydron. Buy one box of 90 lenses.\nBiomedics 1 Day Extra contacts give you all the convenience of a daily disposable lens with no need for solutions, cases or cleaning and are perfect for the occasional wear. These lenses have greater comfort handling with superior ease of insertion and removal.\nBiomedic 1 Day Extra are also marketed under various other brand names including Clear Choice 1-day, Ascend 1-day, easyvision CLARISION SPHERE, Clearsight 1 Day and ProView Daily Disposable.",
            "BANGALORE CY JUNCTION SBC to GONDIA JUNCTION G train timings, routes, stops, and complete info.\nAs of now, 1 trains run between from BANGALORE CY JUNCTION (YPR) to GONDIA JUNCTION (G).\nThe fastest train from BANGALORE CY JUNCTION (YPR) to GONDIA JUNCTION (G) is YPR KRBA WAINGANGA EXP (12251) that departs at 23:40 and arrives to at 21:15. It takes approximately 21:35 hours.",
        ]
        for junk in junk_list:
            self.assertTrue(is_repetition_removal(junk))
        for ok in ok_list:
            self.assertFalse(is_repetition_removal(ok))


if __name__ == '__main__':
    unittest.main()
