"""
Climate Change Analyzer
Advanced analysis of climate data with predictive modeling and impact assessment
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
from datetime import datetime


class ClimateAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()

    def perform_comprehensive_analysis(self, cleaned_data, combined_analysis):
        """Perform comprehensive climate change analysis"""
        print(" Performing Comprehensive Climate Analysis...")

        analysis_results = {}

        # Temperature trend analysis
        if 'temperature' in cleaned_data:
            analysis_results['temperature_trends'] = self.analyze_temperature_trends(cleaned_data['temperature'])

        # Emissions analysis
        if 'emissions' in cleaned_data:
            analysis_results['emissions_analysis'] = self.analyze_emissions_trends(cleaned_data['emissions'])

        # Energy transition analysis
        if 'energy' in cleaned_data:
            analysis_results['energy_transition'] = self.analyze_energy_transition(cleaned_data['energy'])

        # Country clustering
        analysis_results['country_clusters'] = self.perform_country_clustering(combined_analysis)

        # Climate risk assessment
        analysis_results['risk_assessment'] = self.perform_risk_assessment(combined_analysis)

        # Future projections
        analysis_results['future_projections'] = self.generate_future_projections(cleaned_data)

        print(" Comprehensive climate analysis completed")
        return analysis_results

    def analyze_temperature_trends(self, temp_df):
        """Analyze global temperature trends and patterns"""
        print(" Analyzing Global Temperature Trends...")

        results = {}

        # Recent warming (last 50 years)
        recent_data = temp_df[temp_df['year'] >= 1970]
        if len(recent_data) > 0:
            X = np.array(recent_data['year']).reshape(-1, 1)
            y = recent_data['temperature_anomaly']

            model = LinearRegression()
            model.fit(X, y)

            warming_rate = model.coef_[0]  # °C per year
            results['recent_warming_rate'] = warming_rate * 10  # Convert to °C per decade

        # Decadal analysis
        decades = temp_df['decade'].unique()
        decade_analysis = {}

        for decade in sorted(decades):
            decade_data = temp_df[temp_df['decade'] == decade]
            if len(decade_data) > 0:
                avg_anomaly = decade_data['temperature_anomaly'].mean()
                decade_analysis[decade] = {
                    'avg_temperature_anomaly': avg_anomaly,
                    'warming_since_preindustrial': avg_anomaly - temp_df[temp_df['decade'] == 1850][
                        'temperature_anomaly'].mean()
                }

        results['decadal_analysis'] = decade_analysis

        # Critical thresholds analysis
        current_anomaly = temp_df[temp_df['year'] == 2023]['temperature_anomaly'].iloc[0] if len(
            temp_df[temp_df['year'] == 2023]) > 0 else 1.2
        results['current_warming'] = current_anomaly
        results['paris_gap'] = max(0, current_anomaly - 1.5)  # Gap from Paris Agreement

        # Warming acceleration
        recent_20yr = temp_df[temp_df['year'] >= 2000]
        if len(recent_20yr) > 5:
            X_recent = np.array(recent_20yr['year']).reshape(-1, 1)
            y_recent = recent_20yr['temperature_anomaly']

            model_recent = LinearRegression()
            model_recent.fit(X_recent, y_recent)
            recent_rate = model_recent.coef_[0] * 10

            results['acceleration'] = recent_rate - results.get('recent_warming_rate', 0)

        return results

    def analyze_emissions_trends(self, emissions_df):
        """Analyze CO2 emissions trends by country and region"""
        print(" Analyzing CO2 Emissions Trends...")

        results = {}

        # Global emissions trends
        global_emissions = emissions_df.groupby('year')['co2'].sum().reset_index()
        if len(global_emissions) > 0:
            X = np.array(global_emissions['year']).reshape(-1, 1)
            y = global_emissions['co2']

            model = LinearRegression()
            model.fit(X, y)
            results['global_emissions_trend'] = model.coef_[0]  # Annual change

        # Country-level analysis
        country_trends = {}
        latest_year = emissions_df['year'].max()

        for country in emissions_df['country'].unique():
            country_data = emissions_df[emissions_df['country'] == country].sort_values('year')

            if len(country_data) > 5:
                # Recent trend (last 10 years)
                recent_data = country_data[country_data['year'] >= latest_year - 10]
                if len(recent_data) > 2:
                    X_country = np.array(recent_data['year']).reshape(-1, 1)
                    y_country = recent_data['co2']

                    model_country = LinearRegression()
                    model_country.fit(X_country, y_country)

                    trend = model_country.coef_[0]
                    latest_emissions = country_data[country_data['year'] == latest_year]['co2'].iloc[0] if len(
                        country_data[country_data['year'] == latest_year]) > 0 else 0

                    country_trends[country] = {
                        'emissions_trend': trend,
                        'latest_emissions': latest_emissions,
                        'trend_direction': 'Increasing' if trend > 0 else 'Decreasing',
                        'peak_emissions': country_data['co2'].max() == latest_emissions
                    }

        results['country_trends'] = country_trends

        # Development status analysis
        developed = emissions_df[emissions_df['development_status'] == 'Developed']
        developing = emissions_df[emissions_df['development_status'] == 'Developing']

        if len(developed) > 0 and len(developing) > 0:
            results['developed_avg_emissions'] = developed['co2_per_capita'].mean()
            results['developing_avg_emissions'] = developing['co2_per_capita'].mean()
            results['emissions_equity_gap'] = results['developed_avg_emissions'] - results['developing_avg_emissions']

        return results

    def analyze_energy_transition(self, energy_df):
        """Analyze renewable energy transition progress"""
        print(" Analyzing Energy Transition Progress...")

        results = {}

        # Global renewable trends
        global_renewable = energy_df.groupby('year')['renewable_percentage'].mean().reset_index()
        if len(global_renewable) > 0:
            X = np.array(global_renewable['year']).reshape(-1, 1)
            y = global_renewable['renewable_percentage']

            model = LinearRegression()
            model.fit(X, y)
            results['global_renewable_growth'] = model.coef_[0]  # Annual percentage point increase

        # Country transition speed
        transition_speeds = {}
        latest_year = energy_df['year'].max()

        for country in energy_df['country'].unique():
            country_data = energy_df[energy_df['country'] == country].sort_values('year')

            if len(country_data) > 5:
                # Calculate transition speed (last 10 years)
                recent_data = country_data[country_data['year'] >= latest_year - 10]
                if len(recent_data) > 2:
                    speed = (recent_data['renewable_percentage'].iloc[-1] - recent_data['renewable_percentage'].iloc[
                        0]) / len(recent_data)
                    current_level = country_data[country_data['year'] == latest_year]['renewable_percentage'].iloc[
                        0] if len(country_data[country_data['year'] == latest_year]) > 0 else 0

                    transition_speeds[country] = {
                        'transition_speed': speed,
                        'current_renewable': current_level,
                        'years_to_50pct': max(0, (50 - current_level) / speed) if speed > 0 else float('inf'),
                        'transition_phase': 'Advanced' if current_level >= 30 else 'Moderate' if current_level >= 15 else 'Early'
                    }

        results['transition_analysis'] = transition_speeds

        # 2030 projections
        current_global = global_renewable[global_renewable['year'] == latest_year]['renewable_percentage'].iloc[
            0] if len(global_renewable[global_renewable['year'] == latest_year]) > 0 else 20
        growth_rate = results.get('global_renewable_growth', 0.5)
        results['2030_projection'] = current_global + growth_rate * (2030 - latest_year)
        results['paris_alignment'] = results['2030_projection'] >= 50  # Rough alignment indicator

        return results

    def perform_country_clustering(self, analysis_df):
        """Cluster countries based on climate performance"""
        print(" Performing Country Climate Clustering...")

        # Select features for clustering
        features = ['climate_performance_score', 'co2_per_capita', 'renewable_percentage']
        available_features = [f for f in features if f in analysis_df.columns]

        if len(available_features) < 2:
            print("    Not enough features for clustering")
            return analysis_df

        X = analysis_df[available_features].copy()
        X = X.fillna(X.mean())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Determine optimal clusters
        optimal_clusters = self._find_optimal_clusters(X_scaled)

        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        analysis_df['climate_cluster'] = clusters
        analysis_df['cluster_label'] = analysis_df['climate_cluster'].map(
            self._get_climate_cluster_labels(optimal_clusters)
        )

        # Calculate cluster characteristics
        cluster_profiles = {}
        for cluster in analysis_df['climate_cluster'].unique():
            cluster_data = analysis_df[analysis_df['climate_cluster'] == cluster]
            cluster_profiles[cluster] = {
                'avg_score': cluster_data['climate_performance_score'].mean(),
                'avg_emissions': cluster_data['co2_per_capita'].mean(),
                'avg_renewable': cluster_data['renewable_percentage'].mean(),
                'size': len(cluster_data),
                'representative_countries': cluster_data.nlargest(3, 'climate_performance_score')['country'].tolist()
            }

        return {
            'clustered_data': analysis_df,
            'cluster_profiles': cluster_profiles,
            'optimal_clusters': optimal_clusters
        }

    def perform_risk_assessment(self, analysis_df):
        """Assess climate risks for each country"""
        print(" Performing Climate Risk Assessment...")

        risk_data = []

        for _, country in analysis_df.iterrows():
            # Calculate risk score (0-100, higher = more risk)
            emissions_risk = min(100, country['co2_per_capita'] * 5)
            transition_risk = max(0, 100 - country['renewable_percentage'] * 2)
            performance_risk = max(0, 100 - country['climate_performance_score'])

            # Weighted risk score
            overall_risk = (emissions_risk * 0.4 + transition_risk * 0.3 + performance_risk * 0.3)

            # Risk category
            if overall_risk >= 70:
                risk_category = 'Extreme Risk'
            elif overall_risk >= 50:
                risk_category = 'High Risk'
            elif overall_risk >= 30:
                risk_category = 'Moderate Risk'
            else:
                risk_category = 'Low Risk'

            risk_data.append({
                'country': country['country'],
                'overall_risk_score': round(overall_risk, 1),
                'risk_category': risk_category,
                'emissions_risk': round(emissions_risk, 1),
                'transition_risk': round(transition_risk, 1),
                'performance_risk': round(performance_risk, 1),
                'key_vulnerabilities': self._identify_vulnerabilities(country)
            })

        risk_df = pd.DataFrame(risk_data)
        return risk_df

    def generate_future_projections(self, cleaned_data):
        """Generate future climate projections"""
        print(" Generating Future Climate Projections...")

        projections = {}

        # Temperature projections
        if 'temperature' in cleaned_data:
            temp_proj = self._project_temperatures(cleaned_data['temperature'])
            projections['temperature'] = temp_proj

        # Emissions projections
        if 'emissions' in cleaned_data:
            emissions_proj = self._project_emissions(cleaned_data['emissions'])
            projections['emissions'] = emissions_proj

        # Renewable energy projections
        if 'energy' in cleaned_data:
            energy_proj = self._project_renewables(cleaned_data['energy'])
            projections['energy'] = energy_proj

        return projections

    def _project_temperatures(self, temp_df):
        """Project future temperature increases"""
        recent_data = temp_df[temp_df['year'] >= 1950].copy()

        if len(recent_data) < 10:
            return {}

        X = np.array(recent_data['year']).reshape(-1, 1)
        y = recent_data['temperature_anomaly']

        model = LinearRegression()
        model.fit(X, y)

        future_years = [2030, 2050, 2100]
        projections = {}

        for year in future_years:
            projection = model.predict([[year]])[0]
            projections[year] = round(projection, 2)

        return projections

    def _project_emissions(self, emissions_df):
        """Project future emissions based on current trends"""
        projections = {}
        latest_year = emissions_df['year'].max()

        # Global emissions projection
        global_emissions = emissions_df.groupby('year')['co2'].sum().reset_index()

        if len(global_emissions) > 5:
            X = np.array(global_emissions['year']).reshape(-1, 1)
            y = global_emissions['co2']

            model = LinearRegression()
            model.fit(X, y)

            future_years = [2030, 2050]
            for year in future_years:
                projection = model.predict([[year]])[0]
                projections[f'global_emissions_{year}'] = round(projection, 0)

        return projections

    def _project_renewables(self, energy_df):
        """Project renewable energy adoption"""
        global_renewable = energy_df.groupby('year')['renewable_percentage'].mean().reset_index()

        if len(global_renewable) < 5:
            return {}

        X = np.array(global_renewable['year']).reshape(-1, 1)
        y = global_renewable['renewable_percentage']

        model = LinearRegression()
        model.fit(X, y)

        future_years = [2030, 2050]
        projections = {}

        for year in future_years:
            projection = model.predict([[year]])[0]
            projections[f'renewable_{year}'] = round(min(100, projection), 1)

        return projections

    def _find_optimal_clusters(self, X_scaled, max_k=5):
        """Find optimal number of clusters"""
        if len(X_scaled) < 3:
            return 1

        best_k = min(3, len(X_scaled))
        best_score = -1

        for k in range(2, min(max_k + 1, len(X_scaled))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)

                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue

        return best_k

    def _get_climate_cluster_labels(self, n_clusters):
        """Get human-readable cluster labels"""
        labels = {
            2: {0: 'Climate Leaders', 1: 'Climate Laggards'},
            3: {0: 'Climate Champions', 1: 'Moderate Performers', 2: 'High Emitters'},
            4: {0: 'Renewable Pioneers', 1: 'Balanced Transition', 2: 'Carbon Intensive', 3: 'Development Focused'},
            5: {0: 'Global Leaders', 1: 'Advanced Transition', 2: 'Emerging Progress',
                3: 'Development Challenges', 4: 'High Risk Emitters'}
        }

        return labels.get(n_clusters, {i: f'Cluster {i + 1}' for i in range(n_clusters)})

    def _identify_vulnerabilities(self, country_data):
        """Identify key climate vulnerabilities for a country"""
        vulnerabilities = []

        if country_data['co2_per_capita'] > 10:
            vulnerabilities.append("High per-capita emissions")

        if country_data['renewable_percentage'] < 20:
            vulnerabilities.append("Slow renewable transition")

        if country_data['climate_performance_score'] < 50:
            vulnerabilities.append("Poor overall climate performance")

        if country_data['emissions_trend'] == 'growing':
            vulnerabilities.append("Growing emissions trend")

        return vulnerabilities if vulnerabilities else ["Moderate risk profile"]
