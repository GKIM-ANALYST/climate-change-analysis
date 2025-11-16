"""
Climate Change Impact Analysis Platform
Analyzing real-world climate data to address global warming challenges
"""

import warnings

warnings.filterwarnings('ignore')

from src.data_loader import ClimateDataLoader
from src.data_cleaner import ClimateDataCleaner
from src.analyzer import ClimateAnalyzer
from src.visualizer import ClimateVisualizer
import pandas as pd


def main():
    print("""
    CLIMATE CHANGE IMPACT ANALYSIS
    Addressing Global Warming Through Data Analysis
    """)

    try:
        # Step 1: Initialize components
        print("\n STEP 1: Initializing Climate Analysis Engine...")
        loader = ClimateDataLoader()
        cleaner = ClimateDataCleaner()
        analyzer = ClimateAnalyzer()
        visualizer = ClimateVisualizer()

        # Step 2: Load Real Climate Data
        print("\n STEP 2: Loading Real Climate Data...")
        climate_data = loader.load_all_climate_data()

        print("    Loaded datasets:")
        for key, data in climate_data.items():
            if data is not None:
                print(f"      • {key}: {len(data)} records")

        # Step 3: Clean and Enrich Data
        print("\n STEP 3: Cleaning and Enhancing Climate Data...")
        cleaned_data, combined_analysis = cleaner.clean_and_enrich_climate_data(climate_data)

        # Step 4: Perform Climate Analysis
        print("\n STEP 4: Performing Climate Impact Analysis...")
        analysis_results = analyzer.perform_comprehensive_analysis(cleaned_data, combined_analysis)

        # Step 5: Generate Climate Visualizations
        print("\n STEP 5: Creating Climate Action Dashboards...")
        visualizer.generate_all_climate_visualizations(cleaned_data, combined_analysis, analysis_results)

        # Step 6: Present Climate Intelligence
        print("\n" + "=" * 80)
        print(" CLIMATE ACTION INTELLIGENCE REPORT")
        print("=" * 80)

        generate_executive_summary(combined_analysis, analysis_results)
        generate_urgent_actions(combined_analysis)
        generate_success_stories(combined_analysis)

        print(f"\n Climate Analysis Complete! Analyzed {len(combined_analysis)} countries")
        print(" Check the 'outputs' folder for climate dashboards and reports")
        print(" All data saved for climate action planning")

    except Exception as e:
        print(f"\n Error during climate analysis: {e}")
        import traceback
        traceback.print_exc()


def generate_executive_summary(analysis_df, results):
    """Generate climate executive summary"""
    print("\n CLIMATE EXECUTIVE SUMMARY")
    print("-" * 50)

    avg_score = analysis_df['climate_performance_score'].mean()
    critical_countries = len(analysis_df[analysis_df['urgency_level'] == 'Critical Action Needed'])

    print(f"    Global Climate Performance: {avg_score:.0f}/100")
    print(f"    Countries Needing Critical Action: {critical_countries}")
    print(f"    Average Renewable Energy: {analysis_df['renewable_percentage'].mean():.1f}%")
    print(f"    Average CO₂ per capita: {analysis_df['co2_per_capita'].mean():.1f} tons")

    # Top performers
    top_5 = analysis_df.nlargest(5, 'climate_performance_score')
    print(f"\n    CLIMATE LEADERS:")
    for _, country in top_5.iterrows():
        print(
            f"      • {country['country']}: {country['climate_performance_score']:.0f}/100 | {country['recommendation'].split(': ')[0]}")


def generate_urgent_actions(analysis_df):
    """Highlight countries needing urgent action"""
    print("\n URGENT CLIMATE ACTIONS NEEDED")
    print("-" * 50)

    urgent_countries = analysis_df[
        analysis_df['urgency_level'].isin(['Critical Action Needed', 'Significant Improvement Required'])]

    if len(urgent_countries) > 0:
        for _, country in urgent_countries.iterrows():
            print(f"   ️ {country['country']}")
            print(f"      Performance Score: {country['climate_performance_score']:.0f}/100")
            print(f"      CO₂ per capita: {country['co2_per_capita']:.1f} tons")
            print(f"      Renewable Energy: {country['renewable_percentage']:.1f}%")
            print(f"      Action: {country['recommendation']}\n")
    else:
        print("   No countries requiring urgent action identified")


def generate_success_stories(analysis_df):
    """Highlight climate success stories"""
    print("\n CLIMATE SUCCESS STORIES")
    print("-" * 50)

    success_stories = analysis_df[analysis_df['climate_performance_score'] >= 70]

    if len(success_stories) > 0:
        for _, country in success_stories.iterrows():
            print(f"    {country['country']}")
            print(f"      Score: {country['climate_performance_score']:.0f}/100")
            print(f"      Renewable Energy: {country['renewable_percentage']:.1f}%")
            print(f"      Status: {country['recommendation'].split(': ')[1]}\n")
    else:
        print("   No standout success stories yet - more climate action needed globally")


if __name__ == "__main__":
    main()
