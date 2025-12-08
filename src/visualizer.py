"""
Climate Change Visualizer
visualizations for climate data analysis and policy communication
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os


class ClimateVisualizer:
    def __init__(self):
        self.output_path = 'outputs/'
        os.makedirs(self.output_path, exist_ok=True)

        # Climate-focused color schemes
        self.climate_colors = {
            'warming': '#FF6B6B',  # Red for warming
            'cooling': '#4ECDC4',  # Teal for cooling
            'renewable': '#45B7D1',  # Blue for renewable
            'fossil': '#FFA07A',  # Orange for fossil fuels
            'emissions': '#8B4513',  # Brown for emissions
            'safe': '#2E8B57',  # Green for safe levels
            'danger': '#DC143C',  # Crimson for danger
            'warning': '#FFD700'  # Gold for warning
        }

        # Set professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(list(self.climate_colors.values()))

    def generate_all_climate_visualizations(self, cleaned_data, combined_analysis, analysis_results):
        """Generate all climate visualizations"""
        print("\n Generating Complete Climate Visualization Suite...")

        self.create_climate_dashboard(cleaned_data, combined_analysis, analysis_results)
        self.create_temperature_timeline(cleaned_data)
        self.create_emissions_comparison(cleaned_data)
        self.create_energy_transition_chart(cleaned_data)
        self.create_country_cluster_analysis(combined_analysis)
        self.create_risk_assessment_map(analysis_results)
        self.create_climate_action_priorities(combined_analysis)

        print(" All climate visualizations generated successfully!")
        print(" Check the 'outputs' folder for:")
        print("   - climate_dashboard.html (Interactive dashboard)")
        print("   - temperature_timeline.png (Global warming trends)")
        print("   - emissions_comparison.png (Country emissions)")
        print("   - energy_transition.png (Renewable progress)")
        print("   - country_clusters.png (Performance groups)")
        print("   - risk_assessment.html (Risk map)")
        print("   - action_priorities.png (Policy recommendations)")

    def create_climate_dashboard(self, cleaned_data, combined_analysis, analysis_results):
        """Create comprehensive climate dashboard"""
        print(" Creating Climate Action Dashboard...")

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Global Temperature Anomaly Timeline',
                'Country Climate Performance Scores',
                'CO₂ Emissions per Capita Comparison',
                'Renewable Energy Transition Status',
                'Climate Risk Assessment by Country',
                'Energy Transition Speed vs Current Level'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08
        )

        # 1. Temperature Timeline (Top Left)
        if 'temperature' in cleaned_data:
            temp_df = cleaned_data['temperature']
            fig.add_trace(
                go.Scatter(
                    x=temp_df['year'],
                    y=temp_df['temperature_anomaly'],
                    mode='lines+markers',
                    name='Temperature Anomaly',
                    line=dict(color=self.climate_colors['warming'], width=3),
                    hovertemplate='Year: %{x}<br>Anomaly: %{y}°C<extra></extra>'
                ),
                row=1, col=1
            )

            # Add Paris Agreement line
            fig.add_hline(y=1.5, line_dash="dash", line_color="red",
                          annotation_text="Paris Agreement Limit", row=1, col=1)

        # 2. Climate Performance Scores (Top Right)
        performance_sorted = combined_analysis.sort_values('climate_performance_score', ascending=True)
        fig.add_trace(
            go.Bar(
                y=performance_sorted['country'],
                x=performance_sorted['climate_performance_score'],
                orientation='h',
                marker_color=[self.climate_colors['safe'] if x >= 70 else
                              self.climate_colors['warning'] if x >= 50 else
                              self.climate_colors['danger'] for x in performance_sorted['climate_performance_score']],
                name='Climate Performance',
                hovertemplate='%{y}<br>Score: %{x}/100<extra></extra>'
            ),
            row=1, col=2
        )

        # 3. CO2 Emissions per Capita (Middle Left)
        emissions_sorted = combined_analysis.sort_values('co2_per_capita', ascending=True)
        fig.add_trace(
            go.Bar(
                y=emissions_sorted['country'],
                x=emissions_sorted['co2_per_capita'],
                orientation='h',
                marker_color=self.climate_colors['emissions'],
                name='CO₂ per Capita',
                hovertemplate='%{y}<br>Emissions: %{x} tons<extra></extra>'
            ),
            row=2, col=1
        )

        # 4. Renewable Energy Percentage (Middle Right)
        renewable_sorted = combined_analysis.sort_values('renewable_percentage', ascending=True)
        fig.add_trace(
            go.Bar(
                y=renewable_sorted['country'],
                x=renewable_sorted['renewable_percentage'],
                orientation='h',
                marker_color=self.climate_colors['renewable'],
                name='Renewable %',
                hovertemplate='%{y}<br>Renewable: %{x}%<extra></extra>'
            ),
            row=2, col=2
        )

        # 5. Risk Assessment (Bottom Left)
        if 'risk_assessment' in analysis_results:
            risk_df = analysis_results['risk_assessment']
            fig.add_trace(
                go.Scatter(
                    x=risk_df['overall_risk_score'],
                    y=risk_df['country'],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=risk_df['overall_risk_score'],
                        colorscale='RdYlGn_r',  # Red = high risk, Green = low risk
                        showscale=True
                    ),
                    text=risk_df['risk_category'],
                    name='Climate Risk',
                    hovertemplate='%{y}<br>Risk Score: %{x}<br>Category: %{text}<extra></extra>'
                ),
                row=3, col=1
            )

        # 6. Transition Analysis (Bottom Right)
        if 'energy' in cleaned_data:
            energy_df = cleaned_data['energy']
            latest_energy = energy_df[energy_df['year'] == energy_df['year'].max()]

            # Calculate transition speed (simplified)
            transition_data = []
            for country in latest_energy['country'].unique():
                country_data = energy_df[energy_df['country'] == country]
                if len(country_data) > 5:
                    recent_growth = country_data.nlargest(5, 'year')['renewable_percentage'].diff().mean()
                    current_level = country_data.nlargest(1, 'year')['renewable_percentage'].iloc[0]
                    transition_data.append({
                        'country': country,
                        'current_level': current_level,
                        'growth_rate': recent_growth if not pd.isna(recent_growth) else 0
                    })

            if transition_data:
                transition_df = pd.DataFrame(transition_data)
                fig.add_trace(
                    go.Scatter(
                        x=transition_df['current_level'],
                        y=transition_df['growth_rate'],
                        mode='markers+text',
                        marker=dict(
                            size=transition_df['current_level'] / 2,
                            color=transition_df['current_level'],
                            colorscale='Blues',
                            showscale=True
                        ),
                        text=transition_df['country'],
                        textposition="middle center",
                        name='Transition Speed',
                        hovertemplate='%{text}<br>Current: %{x}%<br>Growth: %{y:.1f}%/yr<extra></extra>'
                    ),
                    row=3, col=2
                )

        # Update layout
        fig.update_layout(
            height=1200,
            title_text=" Climate Change Action Dashboard",
            title_x=0.5,
            showlegend=True,
            template="plotly_white"
        )

        # Update axes labels
        fig.update_xaxes(title_text="Temperature Anomaly (°C)", row=1, col=1)
        fig.update_xaxes(title_text="Performance Score", row=1, col=2)
        fig.update_xaxes(title_text="CO₂ per Capita (tons)", row=2, col=1)
        fig.update_xaxes(title_text="Renewable Energy (%)", row=2, col=2)
        fig.update_xaxes(title_text="Risk Score", row=3, col=1)
        fig.update_xaxes(title_text="Current Renewable Level (%)", row=3, col=2)

        fig.update_yaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Countries", row=1, col=2)
        fig.update_yaxes(title_text="Countries", row=2, col=1)
        fig.update_yaxes(title_text="Countries", row=2, col=2)
        fig.update_yaxes(title_text="Countries", row=3, col=1)
        fig.update_yaxes(title_text="Annual Growth Rate (% points)", row=3, col=2)

        fig.write_html(f"{self.output_path}climate_dashboard.html")
        print(" Climate dashboard created: climate_dashboard.html")

    def create_temperature_timeline(self, cleaned_data):
        """Create detailed temperature timeline visualization"""
        print("️ Creating Temperature Timeline...")

        if 'temperature' not in cleaned_data:
            return

        temp_df = cleaned_data['temperature']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 1. Full temperature timeline
        ax1.plot(temp_df['year'], temp_df['temperature_anomaly'],
                 color=self.climate_colors['warming'], linewidth=2.5, label='Temperature Anomaly')
        ax1.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Paris Agreement Limit')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.fill_between(temp_df['year'], temp_df['temperature_anomaly'], 0,
                         alpha=0.3, color=self.climate_colors['warming'])
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Temperature Anomaly (°C)')
        ax1.set_title('Global Temperature Anomaly Timeline (1850-2023)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Recent warming (last 50 years)
        recent_temp = temp_df[temp_df['year'] >= 1970]
        ax2.plot(recent_temp['year'], recent_temp['temperature_anomaly'],
                 color=self.climate_colors['danger'], linewidth=2.5)
        ax2.fill_between(recent_temp['year'], recent_temp['temperature_anomaly'], 0,
                         alpha=0.3, color=self.climate_colors['danger'])
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Temperature Anomaly (°C)')
        ax2.set_title('Recent Global Warming (1970-2023)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add trend line for recent warming
        z = np.polyfit(recent_temp['year'], recent_temp['temperature_anomaly'], 1)
        p = np.poly1d(z)
        ax2.plot(recent_temp['year'], p(recent_temp['year']), "r--", alpha=0.8,
                 label=f'Trend: {z[0] * 10:.2f}°C/decade')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_path}temperature_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(" Temperature timeline created: temperature_timeline.png")

    def create_emissions_comparison(self, cleaned_data):
        """Create emissions comparison charts"""
        print(" Creating Emissions Comparison...")

        if 'emissions' not in cleaned_data:
            return

        emissions_df = cleaned_data['emissions']
        latest_emissions = emissions_df[emissions_df['year'] == emissions_df['year'].max()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Total emissions by country
        country_totals = latest_emissions.groupby('country')['co2'].sum().sort_values(ascending=False)
        bars1 = ax1.bar(range(len(country_totals)), country_totals.values,
                        color=self.climate_colors['emissions'])
        ax1.set_xticks(range(len(country_totals)))
        ax1.set_xticklabels(country_totals.index, rotation=45, ha='right')
        ax1.set_ylabel('CO₂ Emissions (Million Tons)')
        ax1.set_title('Total CO₂ Emissions by Country (Latest Year)', fontweight='bold')

        # Add value labels
        for i, bar in enumerate(bars1):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                     f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)

        # 2. Per capita emissions
        per_capita_sorted = latest_emissions.sort_values('co2_per_capita', ascending=False)
        bars2 = ax2.bar(range(len(per_capita_sorted)), per_capita_sorted['co2_per_capita'],
                        color=self.climate_colors['fossil'])
        ax2.set_xticks(range(len(per_capita_sorted)))
        ax2.set_xticklabels(per_capita_sorted['country'], rotation=45, ha='right')
        ax2.set_ylabel('CO₂ per Capita (Tons)')
        ax2.set_title('CO₂ Emissions per Capita by Country', fontweight='bold')

        # Add value labels
        for i, bar in enumerate(bars2):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{self.output_path}emissions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(" Emissions comparison created: emissions_comparison.png")

    def create_energy_transition_chart(self, cleaned_data):
        """Create energy transition visualization"""
        print(" Creating Energy Transition Chart...")

        if 'energy' not in cleaned_data:
            return

        energy_df = cleaned_data['energy']
        latest_energy = energy_df[energy_df['year'] == energy_df['year'].max()]

        fig, ax = plt.subplots(figsize=(14, 8))

        # Sort by renewable percentage
        sorted_energy = latest_energy.sort_values('renewable_percentage', ascending=True)

        y_pos = np.arange(len(sorted_energy))
        bar_height = 0.35

        # Renewable energy bars
        bars_renewable = ax.barh(y_pos, sorted_energy['renewable_percentage'],
                                 bar_height, label='Renewable Energy',
                                 color=self.climate_colors['renewable'])

        # Fossil fuel bars (remaining percentage)
        bars_fossil = ax.barh(y_pos, sorted_energy['fossil_percentage'],
                              bar_height, left=sorted_energy['renewable_percentage'],
                              label='Fossil Fuels', color=self.climate_colors['fossil'])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_energy['country'])
        ax.set_xlabel('Energy Mix Percentage (%)')
        ax.set_title('Renewable vs Fossil Fuel Energy Mix by Country', fontsize=14, fontweight='bold')
        ax.legend()

        # Add value labels
        for i, (renewable, fossil) in enumerate(zip(sorted_energy['renewable_percentage'],
                                                    sorted_energy['fossil_percentage'])):
            ax.text(renewable / 2, i, f'{renewable:.1f}%', ha='center', va='center',
                    color='white', fontweight='bold')
            ax.text(renewable + fossil / 2, i, f'{fossil:.1f}%', ha='center', va='center',
                    color='white', fontweight='bold')

        # Add 50% target line
        ax.axvline(x=50, color='green', linestyle='--', alpha=0.7, label='50% Target')

        plt.tight_layout()
        plt.savefig(f'{self.output_path}energy_transition.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(" Energy transition chart created: energy_transition.png")

    def create_country_cluster_analysis(self, combined_analysis):
        """Create country clustering visualization"""
        print(" Creating Country Cluster Analysis...")

        if 'climate_cluster' not in combined_analysis.columns:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Cluster distribution
        cluster_counts = combined_analysis['cluster_label'].value_counts()
        wedges, texts, autotexts = ax1.pie(cluster_counts.values, labels=cluster_counts.index,
                                           autopct='%1.1f%%', startangle=90,
                                           colors=[self.climate_colors['safe'],
                                                   self.climate_colors['warning'],
                                                   self.climate_colors['danger'],
                                                   self.climate_colors['emissions']][:len(cluster_counts)])
        ax1.set_title('Country Climate Performance Clusters', fontweight='bold')

        # Improve text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # 2. Cluster characteristics
        cluster_means = combined_analysis.groupby('cluster_label').agg({
            'climate_performance_score': 'mean',
            'co2_per_capita': 'mean',
            'renewable_percentage': 'mean'
        }).round(1)

        # Plot cluster characteristics
        x = np.arange(len(cluster_means))
        width = 0.25

        ax2.bar(x - width, cluster_means['climate_performance_score'], width,
                label='Performance Score', color=self.climate_colors['safe'])
        ax2.bar(x, cluster_means['co2_per_capita'], width,
                label='CO₂ per Capita', color=self.climate_colors['emissions'])
        ax2.bar(x + width, cluster_means['renewable_percentage'], width,
                label='Renewable %', color=self.climate_colors['renewable'])

        ax2.set_xlabel('Climate Clusters')
        ax2.set_ylabel('Scores and Metrics')
        ax2.set_title('Cluster Characteristics Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(cluster_means.index, rotation=45)
        ax2.legend()

        # Add value labels
        for i, (score, emissions, renewable) in enumerate(zip(cluster_means['climate_performance_score'],
                                                              cluster_means['co2_per_capita'],
                                                              cluster_means['renewable_percentage'])):
            ax2.text(i - width, score + 1, f'{score}', ha='center', va='bottom', fontsize=9)
            ax2.text(i, emissions + 0.5, f'{emissions}', ha='center', va='bottom', fontsize=9)
            ax2.text(i + width, renewable + 1, f'{renewable}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{self.output_path}country_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(" Country cluster analysis created: country_clusters.png")

    def create_risk_assessment_map(self, analysis_results):
        """Create climate risk assessment visualization"""
        print(" Creating Risk Assessment Map...")

        if 'risk_assessment' not in analysis_results:
            return

        risk_df = analysis_results['risk_assessment']

        # Create interactive risk map
        fig = px.scatter(risk_df,
                         x='emissions_risk',
                         y='transition_risk',
                         size='overall_risk_score',
                         color='risk_category',
                         hover_name='country',
                         hover_data=['overall_risk_score', 'key_vulnerabilities'],
                         title='Climate Risk Assessment Matrix',
                         color_discrete_map={
                             'Low Risk': self.climate_colors['safe'],
                             'Moderate Risk': self.climate_colors['warning'],
                             'High Risk': self.climate_colors['danger'],
                             'Extreme Risk': '#8B0000'
                         })

        fig.update_layout(
            xaxis_title='Emissions Risk Score',
            yaxis_title='Energy Transition Risk Score',
            showlegend=True
        )

        fig.write_html(f"{self.output_path}risk_assessment.html")
        print(" Risk assessment map created: risk_assessment.html")

    def create_climate_action_priorities(self, combined_analysis):
        """Create climate action priorities matrix"""
        print(" Creating Climate Action Priorities...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create 2x2 matrix: Performance vs Emissions
        for _, country in combined_analysis.iterrows():
            x = country['co2_per_capita']
            y = country['climate_performance_score']

            # Determine quadrant and color
            if y >= 60 and x <= 8:
                color = self.climate_colors['safe']  # High performance, low emissions
                priority = 'Maintain Leadership'
            elif y >= 60 and x > 8:
                color = self.climate_colors['warning']  # High performance, high emissions
                priority = 'Reduce Emissions'
            elif y < 60 and x <= 8:
                color = self.climate_colors['renewable']  # Low performance, low emissions
                priority = 'Improve Overall'
            else:
                color = self.climate_colors['danger']  # Low performance, high emissions
                priority = 'Urgent Action'

            ax.scatter(x, y, s=200, alpha=0.7, color=color)
            ax.annotate(country['country'], (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        # Add quadrant lines
        ax.axhline(y=60, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=8, color='gray', linestyle='--', alpha=0.5)

        # Add quadrant labels
        ax.text(2, 85, 'Climate Leaders\n(Maintain & Support)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))
        ax.text(15, 85, 'High Emitters\n(Reduce Carbon Intensity)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.7))
        ax.text(2, 30, 'Developing Nations\n(Build Green Infrastructure)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))
        ax.text(15, 30, 'Urgent Action Needed\n(Comprehensive Climate Policy)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", alpha=0.7))

        ax.set_xlabel('CO₂ Emissions per Capita (Tons)')
        ax.set_ylabel('Climate Performance Score')
        ax.set_title('Climate Action Priority Matrix', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_path}action_priorities.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Climate action priorities created: action_priorities.png")
