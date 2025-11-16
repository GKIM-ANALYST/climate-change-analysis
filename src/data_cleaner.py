"""
Climate Data Cleaner - FIXED VERSION
Processing and enriching climate change data with robust error handling
"""

import pandas as pd
import numpy as np
from config.settings import IMPACT_CATEGORIES, CLIMATE_GOALS


class ClimateDataCleaner:
    def __init__(self):
        self.processed_data_path = 'data/processed/'

    def clean_and_enrich_climate_data(self, data_dict):
        """Clean and enrich all climate datasets"""
        print(" Cleaning and enriching climate data...")

        cleaned_data = {}

        # Clean each dataset
        if 'temperature' in data_dict:
            cleaned_data['temperature'] = self.clean_temperature_data(data_dict['temperature'])

        if 'emissions' in data_dict:
            cleaned_data['emissions'] = self.clean_emissions_data(data_dict['emissions'])

        if 'energy' in data_dict:
            cleaned_data['energy'] = self.clean_energy_data(data_dict['energy'])

        if 'sea_level' in data_dict:
            cleaned_data['sea_level'] = self.clean_sea_level_data(data_dict['sea_level'])

        # Create combined analysis dataset
        combined_df = self.create_combined_analysis(cleaned_data)

        print(" Climate data cleaning completed")
        return cleaned_data, combined_df

    def clean_temperature_data(self, df):
        """Clean and enrich temperature data"""
        df_clean = df.copy()

        # Add temperature categories
        conditions = [
            df_clean['temperature_anomaly'] <= 0,
            df_clean['temperature_anomaly'] <= 0.5,
            df_clean['temperature_anomaly'] <= 1.0,
            df_clean['temperature_anomaly'] <= 1.5,
            True
        ]
        choices = ['Pre-Industrial', 'Moderate', 'Significant', 'Dangerous', 'Critical']
        df_clean['warming_category'] = np.select(conditions, choices)

        # Calculate warming rates
        df_clean = df_clean.sort_values('year')
        df_clean['warming_rate'] = df_clean['temperature_anomaly'].diff()

        # Add Paris Agreement compliance
        df_clean['paris_compliant'] = df_clean['temperature_anomaly'] <= CLIMATE_GOALS['paris_agreement']

        return df_clean

    def clean_emissions_data(self, df):
        """Clean and enrich emissions data"""
        df_clean = df.copy()

        # Ensure we have required columns
        if 'co2_per_capita' not in df_clean.columns:
            # Estimate co2_per_capita if missing
            df_clean['co2_per_capita'] = df_clean.get('co2', 0) / 100  # Rough estimate

        # Add emissions severity categories
        conditions = [
            df_clean['co2_per_capita'] <= IMPACT_CATEGORIES['emissions_severity']['low'],
            df_clean['co2_per_capita'] <= IMPACT_CATEGORIES['emissions_severity']['moderate'],
            df_clean['co2_per_capita'] <= IMPACT_CATEGORIES['emissions_severity']['high'],
            True
        ]
        choices = ['Low', 'Moderate', 'High', 'Critical']
        df_clean['emissions_severity'] = np.select(conditions, choices)

        # Calculate emissions changes and trends
        df_clean = df_clean.sort_values(['country', 'year'])

        # Calculate percentage change
        df_clean['emissions_change_pct'] = df_clean.groupby('country')['co2'].pct_change() * 100

        # Determine trend based on recent changes
        df_clean = self._calculate_emissions_trends(df_clean)

        # Add development status if missing
        if 'development_status' not in df_clean.columns:
            developed_countries = ['United States', 'Germany', 'United Kingdom', 'Japan', 'Canada', 'Australia',
                                   'France']
            df_clean['development_status'] = df_clean['country'].apply(
                lambda x: 'Developed' if x in developed_countries else 'Developing'
            )

        # Net-zero progress calculations
        current_year = 2023
        df_recent = df_clean[df_clean['year'] == current_year].copy()
        if len(df_recent) > 0:
            df_recent['years_to_netzero'] = CLIMATE_GOALS['carbon_neutrality'] - current_year
            df_recent['required_reduction_pct'] = (df_recent['co2_per_capita'] / df_recent['years_to_netzero'])

        return df_clean

    def _calculate_emissions_trends(self, df):
        """Calculate emissions trends for each country"""
        df_clean = df.copy()

        # Group by country and calculate trend
        trend_data = []
        for country in df_clean['country'].unique():
            country_data = df_clean[df_clean['country'] == country].sort_values('year')

            if len(country_data) > 5:
                # Use last 10 years for trend calculation
                recent_data = country_data[country_data['year'] >= 2013]
                if len(recent_data) > 2:
                    # Calculate linear trend
                    X = np.array(range(len(recent_data))).reshape(-1, 1)
                    y = recent_data['co2'].values

                    try:
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        model.fit(X, y)
                        trend_slope = model.coef_[0]

                        if trend_slope > 0:
                            trend = 'growing'
                        elif trend_slope < 0:
                            trend = 'declining'
                        else:
                            trend = 'stable'
                    except:
                        trend = 'unknown'
                else:
                    trend = 'insufficient_data'
            else:
                trend = 'insufficient_data'

            # Add trend to all rows for this country
            for idx in country_data.index:
                trend_data.append({'index': idx, 'trend': trend})

        # Merge trends back into dataframe
        if trend_data:
            trends_df = pd.DataFrame(trend_data)
            trends_df = trends_df.set_index('index')
            df_clean = df_clean.merge(trends_df, left_index=True, right_index=True, how='left')
        else:
            df_clean['trend'] = 'unknown'

        return df_clean

    def clean_energy_data(self, df):
        """Clean and enrich energy data"""
        df_clean = df.copy()

        # Add renewable energy targets
        df_clean['meets_50pct_target'] = df_clean['renewable_percentage'] >= (CLIMATE_GOALS['renewable_target'] * 100)

        # Calculate transition speed
        df_clean = df_clean.sort_values(['country', 'year'])
        df_clean['renewable_growth'] = df_clean.groupby('country')['renewable_percentage'].diff()

        # Energy transition categories
        conditions = [
            df_clean['renewable_percentage'] >= 50,
            df_clean['renewable_percentage'] >= 30,
            df_clean['renewable_percentage'] >= 15,
            True
        ]
        choices = ['Advanced', 'Moderate', 'Early', 'Lagging']
        df_clean['transition_stage'] = np.select(conditions, choices)

        return df_clean

    def clean_sea_level_data(self, df):
        """Clean and enrich sea level data"""
        df_clean = df.copy()

        # Add impact categories
        conditions = [
            df_clean['sea_level_rise_cm'] <= 10,
            df_clean['sea_level_rise_cm'] <= 20,
            df_clean['sea_level_rise_cm'] <= 30,
            True
        ]
        choices = ['Low Impact', 'Moderate Impact', 'High Impact', 'Severe Impact']
        df_clean['impact_category'] = np.select(conditions, choices)

        # Calculate acceleration
        df_clean = df_clean.sort_values('year')
        df_clean['rise_acceleration'] = df_clean['annual_rise_mm'].diff()

        return df_clean

    def create_combined_analysis(self, cleaned_data):
        """Create comprehensive climate analysis dataset"""
        print("🔗 Creating combined climate analysis...")

        analysis_data = []

        # Analyze each country's climate performance
        if 'emissions' in cleaned_data and 'energy' in cleaned_data:
            emissions_df = cleaned_data['emissions']
            energy_df = cleaned_data['energy']

            # Get latest data for each country (2023)
            latest_year = 2023
            latest_emissions = emissions_df[emissions_df['year'] == latest_year]
            latest_energy = energy_df[energy_df['year'] == latest_year]

            for country in latest_emissions['country'].unique():
                country_emissions = latest_emissions[latest_emissions['country'] == country]
                country_energy = latest_energy[latest_energy['country'] == country]

                if len(country_emissions) > 0 and len(country_energy) > 0:
                    emissions_row = country_emissions.iloc[0]
                    energy_row = country_energy.iloc[0]

                    # Safely get values with defaults
                    co2_per_capita = emissions_row.get('co2_per_capita', 0)
                    renewable_pct = energy_row.get('renewable_percentage', 0)
                    emissions_trend = emissions_row.get('trend', 'unknown')
                    development_status = emissions_row.get('development_status', 'Unknown')

                    # Climate performance score (0-100)
                    emissions_score = max(0, 100 - (co2_per_capita * 3))
                    energy_score = min(100, renewable_pct * 1.5)
                    overall_score = (emissions_score * 0.6 + energy_score * 0.4)

                    # Urgency level
                    if overall_score < 40:
                        urgency = 'Critical Action Needed'
                    elif overall_score < 60:
                        urgency = 'Significant Improvement Required'
                    elif overall_score < 80:
                        urgency = 'Moderate Action Needed'
                    else:
                        urgency = 'On Track'

                    analysis_data.append({
                        'country': country,
                        'climate_performance_score': round(overall_score, 1),
                        'co2_per_capita': co2_per_capita,
                        'renewable_percentage': renewable_pct,
                        'emissions_trend': emissions_trend,
                        'energy_transition_stage': energy_row.get('transition_stage', 'Unknown'),
                        'urgency_level': urgency,
                        'development_status': development_status,
                        'recommendation': self._generate_recommendation(overall_score, co2_per_capita, renewable_pct)
                    })

        analysis_df = pd.DataFrame(analysis_data)

        # Save the analysis
        analysis_df.to_csv(f"{self.processed_data_path}climate_analysis.csv", index=False)
        print(f" Created combined analysis for {len(analysis_df)} countries")

        return analysis_df

    def _generate_recommendation(self, score, emissions, renewable_pct):
        """Generate climate action recommendations"""
        if score < 40:
            if emissions > 15:
                return " CRITICAL: Immediate decarbonization and massive renewable investment needed"
            else:
                return " CRITICAL: Accelerate renewable transition and implement comprehensive climate policy"
        elif score < 60:
            if renewable_pct < 20:
                return "URGENT: Accelerate renewable transition and implement carbon pricing"
            else:
                return " URGENT: Focus on emissions reduction and industrial decarbonization"
        elif score < 80:
            if renewable_pct < 40:
                return " PROGRESSING: Continue renewable expansion and efficiency improvements"
            else:
                return " PROGRESSING: Maintain momentum and address remaining emissions sources"
        else:
            if renewable_pct >= 50:
                return " LEADING: Maintain leadership and support global climate efforts"
            else:
                return " LEADING: Continue excellence and share best practices globally"