import unittest
from calculations import calc_distance, calc_angle, calc_line_orientation, calc_vertex_angle

class TestKeypointHelpers(unittest.TestCase):
    def test_calc_distance(self):
        self.assertAlmostEqual(calc_distance((0,0), (3,4)), 5, places=2)
        self.assertAlmostEqual(calc_distance((1,1), (1,1)), 0, places=2)
        self.assertAlmostEqual(calc_distance((-1,-1), (2,3)), 5, places=2)
        self.assertAlmostEqual(calc_distance((1,2), (1,9)), 7, places=2)

    def test_calc_angle(self):
        self.assertAlmostEqual(calc_angle((0,1), (0,0), (1,0)), 90, places=2)
        self.assertAlmostEqual(calc_angle((0,0), (1,0), (2,0)), 180, places=2)
        self.assertAlmostEqual(calc_angle((1,1), (0,0), (1,0)), 45, places=2)
        self.assertAlmostEqual(calc_angle((-1,1), (0,0), (-1,-1)), 90, places=2)
        self.assertEqual(calc_angle((0,0), (0,0), (1,1)), 0)

    def test_calc_line_orientation(self):
        self.assertAlmostEqual(calc_line_orientation((0,0), (1,0)), 0, places=2)
        self.assertAlmostEqual(calc_line_orientation((0,0), (0,1)), 90, places=2)
        self.assertAlmostEqual(calc_line_orientation((0,0), (-1,0)), 180, places=2)
        self.assertAlmostEqual(calc_line_orientation((0,0), (0,-1)), 270, places=2)
        self.assertAlmostEqual(calc_line_orientation((1,1), (1,1)), 0, places=2)

    def test_calc_vertex_angle(self):
        self.assertAlmostEqual(calc_vertex_angle((-1,0), (0,1), (1,0)), 0, places=2)
        self.assertAlmostEqual(calc_vertex_angle((0, -1), (1, 0), (0, 1)), 90, places=1)
        self.assertAlmostEqual(calc_vertex_angle((0, 1), (-1, 0), (0, -1)), -90, places=1)
        self.assertAlmostEqual(calc_vertex_angle((-1, 0), (0, -1), (1, 0)), 180, places=1)

        

if __name__ == '__main__':
    unittest.main()
