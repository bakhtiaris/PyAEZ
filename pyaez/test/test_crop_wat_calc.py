import unittest
import numpy as np
from CropWatCalc import (
    Adjustkc_Factor,
    eta,
    YieldReductionByWaterDeficit,
    WaterBalance,
    calculateMoistureLimitedYieldNumba
)

class TestCropWatCalc(unittest.TestCase):
    """Detailed test suite for CropWatCalc module"""

    def setUp(self):
        """Set up test data"""
        # Basic parameters
        self.kc = np.array([0.3, 1.2, 0.5])  # Crop coefficients
        self.d_per = np.array([16, 26, 33, 25])  # Stage percentages
        self.cycle_len = 120
        self.yloss_f = np.array([0.4, 0.4, 0.9, 0.5])  # Yield loss factors
        self.yloss_f_all = 1.25
        self.crop_group = 1
        self.plant_height = 1.5
        self.Sa = 100.0  # Soil moisture holding capacity
        self.D1 = 0.3  # Initial rooting depth
        self.D2 = 1.0  # Maximum rooting depth

        # Climate data for 120 days
        days = np.arange(self.cycle_len)
        self.precipitation = np.ones(self.cycle_len) * 5  # mm/day
        self.eto = np.ones(self.cycle_len) * 4  # mm/day
        self.min_temp = np.ones(self.cycle_len) * 20  # Celsius
        self.max_temp = np.ones(self.cycle_len) * 30  # Celsius
        self.mean_temp = (self.min_temp + self.max_temp) / 2
        self.wind_sp = np.ones(self.cycle_len) * 2  # m/s
        self.rel_humidity = np.ones(self.cycle_len) * 0.7
        
        # Create seasonal pattern
        t = np.linspace(0, 2*np.pi, self.cycle_len)
        self.precipitation_seasonal = 5 + 3 * np.sin(t)
        self.eto_seasonal = 4 + 2 * np.sin(t)
        self.min_temp_seasonal = 20 + 5 * np.sin(t)
        self.max_temp_seasonal = 30 + 5 * np.sin(t)
        self.mean_temp_seasonal = (self.min_temp_seasonal + self.max_temp_seasonal) / 2

    def test_eta(self):
        """Test actual evapotranspiration calculation"""
        # Test with different water balance scenarios
        test_cases = [
            # wb_old, etm, Sa, D, p, rain
            (50.0, 5.0, 100.0, 1.0, 0.5, 2.0),  # Normal case
            (10.0, 5.0, 100.0, 1.0, 0.5, 6.0),  # Plenty of rain
            (30.0, 8.0, 100.0, 1.0, 0.5, 0.0),  # No rain
            (80.0, 5.0, 100.0, 1.0, 0.5, 2.0),  # High water balance
        ]
        
        for wb_old, etm, Sa, D, p, rain in test_cases:
            wb, wx, eta_val = eta(wb_old, etm, Sa, D, p, rain)
            
            # Check results
            self.assertTrue(0 <= wb <= Sa * D)  # Water balance should be within bounds
            self.assertTrue(wx >= 0)  # Excess water should be non-negative
            self.assertTrue(0 <= eta_val <= etm)  # ETa should be between 0 and ETm
            
            # Water balance + excess water + ETa should equal old water balance + rain
            self.assertAlmostEqual(wb + wx + eta_val, wb_old + rain, places=5)

    def test_Adjustkc_Factor(self):
        """Test kc factor adjustment"""
        kc1, kc2, kc3 = Adjustkc_Factor(
            self.kc,
            self.d_per,
            self.cycle_len,
            self.precipitation,
            self.eto,
            self.min_temp,
            self.max_temp,
            self.plant_height,
            self.wind_sp,
            'R'  # Rainfed
        )
        
        # Check kc values are reasonable
        self.assertTrue(0 < kc1 < 1.5)
        self.assertTrue(0 < kc2 < 2.0)
        self.assertTrue(0 < kc3 < 1.5)
        
        # Test with irrigated conditions
        kc1_irr, kc2_irr, kc3_irr = Adjustkc_Factor(
            self.kc,
            self.d_per,
            self.cycle_len,
            self.precipitation,
            self.eto,
            self.min_temp,
            self.max_temp,
            self.plant_height,
            self.wind_sp,
            'I'  # Irrigated
        )
        
        # Check kc values are reasonable
        self.assertTrue(0 < kc1_irr < 1.5)
        self.assertTrue(0 < kc2_irr < 2.0)
        self.assertTrue(0 < kc3_irr < 1.5)
        
        # Initial kc may differ between irrigated and rainfed due to wet events handling
        self.assertAlmostEqual(kc1, kc1_irr, delta=0.5)
 
    def test_YieldReductionByWaterDeficit(self):
        eta_optimal = np.ones(self.cycle_len) * 4.0
        etm = np.ones(self.cycle_len) * 4.0

        # No deficit should result in fc2 exactly 1.0
        fc2_optimal = YieldReductionByWaterDeficit(
            self.yloss_f, eta_optimal, etm, self.cycle_len, self.d_per)
        self.assertAlmostEqual(fc2_optimal, 1.0, places=5)

        # 20% deficit in stage 1 (yloss_f=0.4)
        eta_deficit_stage1 = eta_optimal.copy()
        stage1_end = int(self.cycle_len * self.d_per[0] / 100)
        eta_deficit_stage1[:stage1_end] *= 0.8
        fc2_deficit_stage1 = YieldReductionByWaterDeficit(
            self.yloss_f, eta_deficit_stage1, etm, self.cycle_len, self.d_per)
        expected_fc2_stage1 = 1 - 0.4 * 0.2  # = 0.92
        self.assertAlmostEqual(fc2_deficit_stage1, expected_fc2_stage1, delta=0.01)

        # Severe deficit (50%) in reproductive stage (yloss_f=0.9)
        eta_deficit_stage3 = eta_optimal.copy()
        stage3_start = int(self.cycle_len * sum(self.d_per[:2]) / 100)
        stage3_end = int(self.cycle_len * sum(self.d_per[:3]) / 100)
        eta_deficit_stage3[stage3_start:stage3_end] *= 0.5
        fc2_deficit_stage3 = YieldReductionByWaterDeficit(
            self.yloss_f, eta_deficit_stage3, etm, self.cycle_len, self.d_per)
        expected_fc2_stage3 = 1 - 0.9 * 0.5  # = 0.55
        self.assertAlmostEqual(fc2_deficit_stage3, expected_fc2_stage3, delta=0.01)

        # Deficit across all stages (30%)
        eta_deficit_all = eta_optimal.copy() * 0.7
        fc2_deficit_all = YieldReductionByWaterDeficit(
            self.yloss_f, eta_deficit_all, etm, self.cycle_len, self.d_per)
        expected_fc2_all = min([
            1 - 0.4 * 0.3,  # stage 1
            1 - 0.4 * 0.3,  # stage 2
            1 - 0.9 * 0.3,  # stage 3 (most sensitive, expected lowest)
            1 - 0.5 * 0.3   # stage 4
        ])  # = 0.73 (stage 3 most limiting)
        self.assertAlmostEqual(fc2_deficit_all, expected_fc2_all, delta=0.01)

        

 
    def test_WaterBalance(self):
        Sb_cycle, Wx_cycle, Wb_cycle, Eta_cycle, Etm_cycle, kc_daily, pc_daily = WaterBalance(
            self.eto_seasonal, self.kc, self.d_per, self.cycle_len, self.Sa, self.D1, self.D2,
            self.mean_temp_seasonal, self.max_temp_seasonal, self.precipitation_seasonal, self.crop_group)

        self.assertEqual(Wb_cycle.shape, (self.cycle_len,))
        self.assertTrue(np.all(Wb_cycle >= 0))
        self.assertTrue(np.all(Wb_cycle <= self.Sa))

        correlation = np.corrcoef(self.precipitation_seasonal, Wb_cycle)[0, 1]
        print(f"Water balance vs. precipitation correlation: {correlation}")
        
        # Allowing a realistic negative correlation, given seasonal evapotranspiration dynamics
        self.assertGreater(correlation, -0.6)
        
        

    def test_calculateMoistureLimitedYieldNumba(self):
        """Test moisture limited yield calculation"""
        # Test rainfed condition
        y_potential = 5000  # kg/ha
        
        wde, fc2, eta_total, y_limited = calculateMoistureLimitedYieldNumba(
            'R',  # Rainfed
            self.kc,
            self.d_per,
            self.cycle_len,
            self.precipitation,
            self.eto,
            self.min_temp,
            self.max_temp,
            self.plant_height,
            self.wind_sp,
            self.Sa,
            self.D1,
            self.D2,
            self.mean_temp,
            self.crop_group,
            self.yloss_f_all,
            self.yloss_f,
            False,  # Non-perennial
            y_potential
        )
        
        # Check output types and ranges
        self.assertTrue(wde >= 0)  # Water deficit should be non-negative
        self.assertTrue(0 <= fc2 <= 1)  # fc2 should be between 0 and 1
        self.assertTrue(eta_total >= 0)  # Total ETa should be non-negative
        self.assertTrue(y_limited <= y_potential)  # Limited yield should not exceed potential
        
        # Test irrigated condition
        wde_irr, fc2_irr, eta_total_irr, y_limited_irr = calculateMoistureLimitedYieldNumba(
            'I',  # Irrigated
            self.kc,
            self.d_per,
            self.cycle_len,
            self.precipitation,
            self.eto,
            self.min_temp,
            self.max_temp,
            self.plant_height,
            self.wind_sp,
            self.Sa,
            self.D1,
            self.D2,
            self.mean_temp,
            self.crop_group,
            self.yloss_f_all,
            self.yloss_f,
            False,  # Non-perennial
            y_potential
        )
        
        # For irrigated conditions, fc2 should be 1.0 (no water limitation)
        self.assertEqual(fc2_irr, 1.0)
        self.assertEqual(y_limited_irr, y_potential)
        
        # Water deficit should be higher for irrigated conditions (difference between ETm and ETa)
        self.assertGreaterEqual(wde_irr, wde)
        
        # Test with perennial crop
        wde_per, fc2_per, eta_total_per, y_limited_per = calculateMoistureLimitedYieldNumba(
            'R',  # Rainfed
            self.kc,
            self.d_per,
            self.cycle_len,
            self.precipitation,
            self.eto,
            self.min_temp,
            self.max_temp,
            self.plant_height,
            self.wind_sp,
            self.Sa,
            self.D1,
            self.D2,
            self.mean_temp,
            self.crop_group,
            self.yloss_f_all,
            self.yloss_f,
            True,  # Perennial
            y_potential
        )
        
        # For perennial rainfed crops, fc2 calculation differs
        self.assertLessEqual(fc2_per, 1.0)
        self.assertLessEqual(y_limited_per, y_potential)
        
    def test_various_precipitation_scenarios(self):
        """Test with different precipitation scenarios"""
        # Base potential yield
        y_potential = 5000  # kg/ha
        
        # Various precipitation scenarios
        scenarios = [
            ('Very dry', np.ones(self.cycle_len) * 1.0),  # Very dry
            ('Dry', np.ones(self.cycle_len) * 2.0),  # Dry
            ('Normal', np.ones(self.cycle_len) * 5.0),  # Normal
            ('Wet', np.ones(self.cycle_len) * 8.0),  # Wet
            ('Very wet', np.ones(self.cycle_len) * 12.0)  # Very wet
        ]
        
        results = []
        
        for name, precip in scenarios:
            wde, fc2, eta_total, y_limited = calculateMoistureLimitedYieldNumba(
                'R',  # Rainfed
                self.kc,
                self.d_per,
                self.cycle_len,
                precip,
                self.eto,
                self.min_temp,
                self.max_temp,
                self.plant_height,
                self.wind_sp,
                self.Sa,
                self.D1,
                self.D2,
                self.mean_temp,
                self.crop_group,
                self.yloss_f_all,
                self.yloss_f,
                False,  # Non-perennial
                y_potential
            )
            
            results.append((name, wde, fc2, eta_total, y_limited))
        
        # Check that yields increase with more precipitation
        yields = [r[4] for r in results]
        self.assertTrue(all(yields[i] <= yields[i+1] for i in range(len(yields)-1)))
        
        # Check that water deficits decrease with more precipitation
        deficits = [r[1] for r in results]
        self.assertTrue(all(deficits[i] >= deficits[i+1] for i in range(len(deficits)-1)))
        
        # Check that ETa increases with more precipitation
        etas = [r[3] for r in results]
        self.assertTrue(all(etas[i] <= etas[i+1] for i in range(len(etas)-1)))


if __name__ == '__main__':
    unittest.main()