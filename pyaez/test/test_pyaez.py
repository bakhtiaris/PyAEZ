import unittest
import numpy as np
import math
from pyaez.BioMassCalc import (
    calculateBiomassNumba, 
    DayTimeTemperature,
    calculateBiomassNumbaIntermediates
)

class TestBioMassCalc(unittest.TestCase):
    """Detailed test suite for BioMassCalc module"""

    def setUp(self):
        """Set up test data"""
        # Basic parameters
        self.cycle_begin = 1
        self.cycle_end = 121
        self.cycle_len = 120
        self.latitude = 15.0
        self.LAi = 3.0
        self.legume = 0
        self.adaptability = 4
        self.leap_year = False
        
        # Climate data for 120 days
        self.shortRad_daily = np.ones(120) * 200  # W/m2
        self.meanT_daily = np.ones(120) * 25      # Celsius
        self.minT_daily = np.ones(120) * 20       # Celsius
        self.maxT_daily = np.ones(120) * 30       # Celsius
        
        # Create seasonal pattern
        t = np.linspace(0, 2*np.pi, 120)
        self.meanT_daily_seasonal = 25 + 5 * np.sin(t)
        self.minT_daily_seasonal = 20 + 5 * np.sin(t)
        self.maxT_daily_seasonal = 30 + 5 * np.sin(t)
        self.shortRad_daily_seasonal = 200 + 50 * np.sin(t)

    def test_DayTimeTemperature(self):
        """Test day time temperature calculation with constant temperatures"""
        day_temp = DayTimeTemperature(
            self.minT_daily, 
            self.maxT_daily, 
            self.cycle_begin, 
            self.cycle_end, 
            self.leap_year
        )
        
        # Check output shape
        self.assertEqual(day_temp.shape, self.minT_daily.shape)
        
        # Day temperature should be between min and max, but closer to max
        for i in range(len(day_temp)):
            self.assertTrue(self.minT_daily[i] <= day_temp[i] <= self.maxT_daily[i])
            # Day temperature is typically weighted more towards max temperature
            self.assertGreater(day_temp[i] - self.minT_daily[i], self.maxT_daily[i] - day_temp[i])
    
    def test_DayTimeTemperature_seasonal(self):
        """Test day time temperature calculation with seasonal temperature pattern"""
        day_temp = DayTimeTemperature(
            self.minT_daily_seasonal, 
            self.maxT_daily_seasonal, 
            self.cycle_begin, 
            self.cycle_end, 
            self.leap_year
        )
        
        # Check output shape
        self.assertEqual(day_temp.shape, self.minT_daily_seasonal.shape)
        
        # Day temperature should follow seasonal pattern
        for i in range(len(day_temp)):
            self.assertTrue(
                self.minT_daily_seasonal[i] <= day_temp[i] <= self.maxT_daily_seasonal[i]
            )
    
    def test_calculateBiomassNumba_constant(self):
        """Test biomass calculation with constant climate data"""
        bn = calculateBiomassNumba(
            self.cycle_begin, 
            self.cycle_end, 
            self.cycle_len, 
            self.latitude,
            self.shortRad_daily, 
            self.meanT_daily, 
            self.minT_daily, 
            self.maxT_daily,
            self.LAi, 
            self.legume, 
            self.adaptability, 
            self.leap_year
        )
        
        # Check result is reasonable
        self.assertGreater(bn, 0)  # Biomass should be positive
        self.assertLess(bn, 50000)  # And within reasonable bounds
        
        # Test with a different LAI
        bn_higher_lai = calculateBiomassNumba(
            self.cycle_begin, 
            self.cycle_end, 
            self.cycle_len, 
            self.latitude,
            self.shortRad_daily, 
            self.meanT_daily, 
            self.minT_daily, 
            self.maxT_daily,
            self.LAi * 1.5,  # Higher LAI
            self.legume, 
            self.adaptability, 
            self.leap_year
        )
        
        # Higher LAI should produce more biomass
        self.assertGreater(bn_higher_lai, bn)
    
    def test_calculateBiomassNumba_seasonal(self):
        """Test biomass calculation with seasonal climate data"""
        bn = calculateBiomassNumba(
            self.cycle_begin, 
            self.cycle_end, 
            self.cycle_len, 
            self.latitude,
            self.shortRad_daily_seasonal, 
            self.meanT_daily_seasonal, 
            self.minT_daily_seasonal, 
            self.maxT_daily_seasonal,
            self.LAi, 
            self.legume, 
            self.adaptability, 
            self.leap_year
        )
        
        # Check result is reasonable
        self.assertGreater(bn, 0)  # Biomass should be positive
        self.assertLess(bn, 50000)  # And within reasonable bounds
    
    def test_calculateBiomassNumba_legume(self):
        """Test biomass calculation with legume vs non-legume"""
        bn_non_legume = calculateBiomassNumba(
            self.cycle_begin, 
            self.cycle_end, 
            self.cycle_len, 
            self.latitude,
            self.shortRad_daily, 
            self.meanT_daily, 
            self.minT_daily, 
            self.maxT_daily,
            self.LAi, 
            0,  # non-legume
            self.adaptability, 
            self.leap_year
        )
        
        bn_legume = calculateBiomassNumba(
            self.cycle_begin, 
            self.cycle_end, 
            self.cycle_len, 
            self.latitude,
            self.shortRad_daily, 
            self.meanT_daily, 
            self.minT_daily, 
            self.maxT_daily,
            self.LAi, 
            1,  # legume
            self.adaptability, 
            self.leap_year
        )
        
        # Legumes typically have different biomass production due to N fixation
        self.assertNotEqual(bn_non_legume, bn_legume)
    
    def test_calculateBiomassNumba_adaptability(self):
        """Test biomass calculation with different adaptability classes"""
        adaptability_classes = [1, 2, 3, 4]
        biomass_results = []
        
        for adaptability in adaptability_classes:
            bn = calculateBiomassNumba(
                self.cycle_begin, 
                self.cycle_end, 
                self.cycle_len, 
                self.latitude,
                self.shortRad_daily, 
                self.meanT_daily, 
                self.minT_daily, 
                self.maxT_daily,
                self.LAi, 
                self.legume, 
                adaptability, 
                self.leap_year
            )
            biomass_results.append(bn)
        
        # Different adaptability classes should yield different biomass
        self.assertNotEqual(biomass_results[0], biomass_results[-1])
    
    def test_calculateBiomassNumbaIntermediates(self):
        """Test intermediate values from biomass calculation"""
        result = calculateBiomassNumbaIntermediates(
            self.cycle_begin, 
            self.cycle_end, 
            self.cycle_len, 
            self.latitude,
            self.shortRad_daily, 
            self.meanT_daily, 
            self.minT_daily, 
            self.maxT_daily,
            self.LAi, 
            self.legume, 
            self.adaptability, 
            self.leap_year
        )
        
        # Check we get the expected number of return values
        self.assertEqual(len(result), 16)
        
        # Extract individual components
        bn, Ac_mean, bc_mean, bo_mean, meanT_mean, dT_mean, Rg, f_day_clouded, iPm, Ct, l, bgm, Bn, Ac_interp, Bc_interp, Bo_interp = result
        
        # Check output types and ranges
        self.assertIsInstance(bn, float)
        self.assertIsInstance(Ac_mean, float)
        self.assertIsInstance(f_day_clouded, float)
        
        # Day fraction clouded should be between 0 and 1
        self.assertTrue(0 <= f_day_clouded <= 1)
        
        # Mean temperature should match input
        self.assertAlmostEqual(meanT_mean, np.mean(self.meanT_daily))
        
        # Growth rate multiplier should be between 0 and 1.1 for typical LAI values
        self.assertTrue(0 <= l <= 1.1)
        
        # The final biomass value should match direct calculation
        direct_bn = calculateBiomassNumba(
            self.cycle_begin, 
            self.cycle_end, 
            self.cycle_len, 
            self.latitude,
            self.shortRad_daily, 
            self.meanT_daily, 
            self.minT_daily, 
            self.maxT_daily,
            self.LAi, 
            self.legume, 
            self.adaptability, 
            self.leap_year
        )
        self.assertAlmostEqual(bn, direct_bn)


if __name__ == '__main__':
    unittest.main()
