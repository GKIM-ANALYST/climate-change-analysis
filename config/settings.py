"""
Climate Change Analysis Configuration
Real-world data sources and analysis parameters
"""

# Data Sources
DATA_SOURCES = {
    'global_temperature': 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv',
    'co2_emissions': 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv',
    'sea_level': 'https://gml.noaa.gov/ccgg/trends/data.html',
    'renewable_energy': 'https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv'
}

# Countries for Analysis
COUNTRIES = {
    'major_emitters': ['United States', 'China', 'India', 'Russia', 'Japan', 'Germany', 'Iran', 'South Korea', 'Saudi Arabia', 'Canada'],
    'european': ['United Kingdom', 'France', 'Italy', 'Spain', 'Poland', 'Netherlands', 'Belgium', 'Sweden', 'Norway'],
    'developing': ['Brazil', 'Indonesia', 'Mexico', 'South Africa', 'Nigeria', 'Vietnam', 'Philippines', 'Bangladesh']
}

# Analysis Timeframes
TIME_PERIODS = {
    'historical_start': 1850,
    'modern_start': 1950,
    'recent_start': 1990,
    'current_end': 2025
}

# Climate Impact Categories
IMPACT_CATEGORIES = {
    'temperature_increase': {
        'low': 1.0,
        'moderate': 1.5,
        'high': 2.0,
        'critical': 3.0
    },
    'emissions_severity': {
        'low': 5.0,      # tons CO2 per capita
        'moderate': 10.0,
        'high': 15.0,
        'critical': 20.0
    }
}

# Climate Goals
CLIMATE_GOALS = {
    'paris_agreement': 1.5,  # °C
    'carbon_neutrality': 2050,
    'renewable_target': 0.50  # 50% renewable energy
}