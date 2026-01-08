#!/usr/bin/env python3
"""
index.py - GitHub License Popularity Analysis

Python script to sample GitHub repositories, collect metadata (stars, license,
forks, issues, commits, subscribers_count, created_at, owner_type, language, size),
classify license restrictiveness, clean & normalise data, and run a multiple linear
regression to isolate the effect of license type on repository popularity.

Key Features:
 - Random sampling by repository ID for broad distribution
 - License classification into: permissive, weak_copyleft, strong_copyleft, other, no_license
 - Top K programming language dummy variables for controlling language effects
 - Standardized regression coefficients for comparable effect sizes
 - Publication-quality visualizations (PNG and PDF) for thesis inclusion

Usage:
  1. Install dependencies:
     pip install PyGithub python-dotenv pandas scikit-learn matplotlib seaborn

  2. Create a .env file with GITHUB_TOKEN=your_personal_access_token (recommended)

  3. Run (sample new data):
     python index.py --licensed 500 --output data.csv

  4. Or regenerate analysis from existing data:
     python index.py --input existing_data.csv

Output Files:
 - {output}.csv - Raw sampled repository data
 - {output}_cleaned.json - Processed data with dummy variables
 - {output}_other_licenses.json - Details of unclassified license types
 - {output}_regression_results.txt - Full regression analysis results
 - {output}_figures/ - Directory containing visualizations

Notes:
 - GitHub rate limits apply. Use a personal access token to raise rate limits.
 - The sampling method uses random repository IDs for broad distribution.
 - Commit counts are estimated via the commits endpoint (capped at 1000).
 - Baseline categories (excluded from model to avoid dummy variable trap):
   - License baseline: no_license
   - Language baseline: most common language in sample
"""

import os
import json
import time
import argparse
import csv
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv
from github import Github, GithubException
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

# --- License classification map (exact SPDX identifiers) ---
LICENSE_CATEGORY = {
    # Permissive licenses
    'MIT': 'permissive',
    'MIT-0': 'permissive',
    'Apache-2.0': 'permissive',
    'BSD-2-Clause': 'permissive',
    'BSD-3-Clause': 'permissive',
    'BSD-3-Clause-Clear': 'permissive',
    '0BSD': 'permissive',
    'ISC': 'permissive',
    'Unlicense': 'permissive',
    'CC0-1.0': 'permissive',
    'CC-BY-4.0': 'permissive',
    'BSL-1.0': 'permissive',
    'Zlib': 'permissive',
    'WTFPL': 'permissive',
    'MulanPSL-2.0': 'permissive',
    'UPL-1.0': 'permissive',
    'Artistic-2.0': 'permissive',

    # Weak copyleft licenses
    'LGPL-2.1': 'weak_copyleft',
    'LGPL-3.0': 'weak_copyleft',
    'MPL-2.0': 'weak_copyleft',
    'EPL-1.0': 'weak_copyleft',
    'EPL-2.0': 'weak_copyleft',
    'OFL-1.1': 'weak_copyleft',
    'OSL-3.0': 'weak_copyleft',

    # Strong copyleft licenses
    'GPL-2.0': 'strong_copyleft',
    'GPL-3.0': 'strong_copyleft',
    'AGPL-3.0': 'strong_copyleft',
    'CC-BY-SA-4.0': 'strong_copyleft',
    'EUPL-1.2': 'strong_copyleft',
}

def classify_license(spdx):
    """
    Classify license by exact SPDX identifier into restrictiveness categories.
    
    Args:
        spdx: SPDX license identifier string (e.g., 'MIT', 'GPL-3.0')
    
    Returns:
        str: One of 'permissive', 'weak_copyleft', 'strong_copyleft', 'other', or 'no_license'
    """
    if not spdx or spdx == 'NOASSERTION':
        return 'no_license'
    
    return LICENSE_CATEGORY.get(spdx, 'other')


def estimate_commit_count(repo):
    """
    Estimate total commit count for a repository.
    
    Uses PyGithub's totalCount when available, otherwise counts commits
    up to API pagination limits (typically 1000 max).
    
    Args:
        repo: PyGithub Repository object
    
    Returns:
        int or None: Estimated commit count, or None if unavailable
    """
    try:
        commits = repo.get_commits()
        # Try to get total count (this may not always work)
        try:
            return commits.totalCount
        except:
            # Fallback: count up to 1000 commits (API limitation)
            count = 0
            for _ in commits:
                count += 1
                if count >= 1000:
                    break
            return count
    except GithubException:
        return None

def fetch_repo_by_id(g, repo_id):
    """Fetch a repository by its numeric ID, return None if not available"""
    try:
        repo = g.get_repo(repo_id)
        # Skip private, fork, archived, or disabled repos
        if repo.private or repo.fork or repo.archived:
            return None
        if hasattr(repo, 'disabled') and repo.disabled:
            return None
        return repo
    except GithubException:
        return None
    except Exception:
        return None

def fetch_repo_data_from_obj(repo):
    """Extract metadata from a repository object"""
    try:
        # Get license info
        license_spdx = None
        license_name = None
        try:
            if repo.license:
                license_spdx = repo.license.spdx_id
                license_name = repo.license.name
        except:
            pass
        
        # Estimate commit count
        commit_count = estimate_commit_count(repo)
        
        return {
            'full_name': repo.full_name,
            'name': repo.name,
            'owner': repo.owner.login,
            'owner_type': repo.owner.type,
            'stars': repo.stargazers_count,
            'forks_count': repo.forks_count,
            'open_issues_count': repo.open_issues_count,
            'subscribers_count': repo.subscribers_count,
            'size': repo.size,
            'language': repo.language,
            'license_spdx': license_spdx,
            'license_name': license_name,
            'created_at': repo.created_at.isoformat() if repo.created_at else None,
            'archived': repo.archived,
            'disabled': repo.disabled if hasattr(repo, 'disabled') else False,
            'pushed_at': repo.pushed_at.isoformat() if repo.pushed_at else None,
            'commit_count': commit_count
        }
    except GithubException as e:
        print(f'fetchRepoData error for {repo.full_name}: {e.status} {e.data.get("message", "")}')
        return None
    except Exception as e:
        print(f'fetchRepoData error: {str(e)}')
        return None

# Maximum known GitHub repository ID (approximate upper bound for random sampling)
# This value should be periodically updated as GitHub grows
MAX_REPO_ID = 1109900000

def sample_repositories(g, target_licensed, max_unlicensed_ratio=2.0):
    """
    Sample repositories by random ID until we have target_licensed repos with licenses.
    Also collects unlicensed repos up to max_unlicensed_ratio * target_licensed.
    
    Args:
        g: GitHub client
        target_licensed: Target number of repositories WITH licenses
        max_unlicensed_ratio: Maximum ratio of unlicensed to licensed repos to collect
    
    Returns:
        List of repository data dictionaries
    """
    licensed_repos = []
    unlicensed_repos = []
    seen_ids = set()
    max_unlicensed = int(target_licensed * max_unlicensed_ratio)
    
    attempts = 0
    max_attempts = target_licensed * 100  # Allow many attempts since most IDs are invalid
    
    print(f'Sampling repositories by random ID...')
    print(f'Target: {target_licensed} licensed repositories')
    print(f'Max unlicensed: {max_unlicensed} repositories')
    
    while len(licensed_repos) < target_licensed and attempts < max_attempts:
        attempts += 1
        
        # Generate a random repository ID
        repo_id = random.randint(1, MAX_REPO_ID)
        
        if repo_id in seen_ids:
            continue
        seen_ids.add(repo_id)
        
        # Fetch the repository
        repo = fetch_repo_by_id(g, repo_id)
        if not repo:
            continue
        
        # Get full metadata
        data = fetch_repo_data_from_obj(repo)
        if not data:
            continue
        
        # Classify by license status
        has_license = data['license_spdx'] and data['license_spdx'] != 'NOASSERTION'
        
        if has_license:
            licensed_repos.append(data)
            if len(licensed_repos) % 10 == 0:
                print(f'Progress: {len(licensed_repos)}/{target_licensed} licensed, {len(unlicensed_repos)} unlicensed')
        elif len(unlicensed_repos) < max_unlicensed:
            unlicensed_repos.append(data)
            print(f'Progress: {len(licensed_repos)}/{target_licensed} licensed, {len(unlicensed_repos)} unlicensed')
        
        # Rate limiting
        if attempts % 100 == 0:
            time.sleep(1)  # Brief pause every 100 attempts
        else:
            time.sleep(0.1)  # Small delay between requests
    
    if len(licensed_repos) < target_licensed:
        print(f'Warning: Only found {len(licensed_repos)} licensed repos after {attempts} attempts')
    
    print(f'\nSampling complete:')
    print(f'  Licensed repositories: {len(licensed_repos)}')
    print(f'  Unlicensed repositories: {len(unlicensed_repos)}')
    print(f'  Total: {len(licensed_repos) + len(unlicensed_repos)}')
    print(f'  Attempts made: {attempts}')
    
    return licensed_repos + unlicensed_repos

def days_since(date_str):
    """
    Calculate the number of days elapsed since a given ISO date string.
    
    Args:
        date_str: ISO format date string (e.g., '2020-01-15T12:00:00Z')
    
    Returns:
        int: Number of days since the given date, or 0 if date_str is None/empty
    """
    if not date_str:
        return 0
    then = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    now = datetime.now(then.tzinfo) if then.tzinfo else datetime.now()
    return (now - then).days

def main():
    parser = argparse.ArgumentParser(description='Sample GitHub repositories and analyze license popularity')
    parser.add_argument('-t', '--token', help='GitHub personal access token (or set GITHUB_TOKEN)')
    parser.add_argument('-l', '--licensed', type=int, default=500, help='Target number of licensed repositories to sample')
    parser.add_argument('--unlicensed-ratio', type=float, default=2.0, help='Max ratio of unlicensed to licensed repos (default: 2.0)')
    parser.add_argument('-o', '--output', default='github_repos_sample.csv', help='CSV output filename')
    parser.add_argument('-i', '--input', help='Input CSV file from previous run (skips sampling, generates visuals only)')
    parser.add_argument('--language-topk', type=int, default=10, help='How many top languages to keep as dummies')
    
    args = parser.parse_args()
    print('Options:', vars(args))
    
    token = args.token or os.getenv('GITHUB_TOKEN') or ''
    if not token:
        print('Warning: No GitHub token supplied. You will be rate-limited heavily.')
    
    target_licensed = args.licensed
    unlicensed_ratio = args.unlicensed_ratio
    output = args.output
    language_topk = args.language_topk
    input_file = args.input
    
    # If input file provided, load existing data and skip to analysis
    if input_file:
        print(f'Loading data from previous run: {input_file}')
        
        # Read CSV file
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Convert numeric fields back from strings
        for r in rows:
            r['stars'] = int(r['stars']) if r['stars'] else 0
            r['forks_count'] = int(r['forks_count']) if r['forks_count'] else 0
            r['open_issues_count'] = int(r['open_issues_count']) if r['open_issues_count'] else 0
            r['subscribers_count'] = int(r['subscribers_count']) if r['subscribers_count'] else 0
            r['size'] = int(r['size']) if r['size'] else 0
            r['commit_count'] = int(r['commit_count']) if r['commit_count'] else None
        
        print(f'Loaded {len(rows)} repositories from {input_file}')
        
        # Use input file base name for outputs
        output = input_file
    else:
        # Initialize GitHub client and sample repositories
        g = Github(token if token else None)
        
        print(f'Starting sampling. Target: {target_licensed} licensed repositories')
        rows = sample_repositories(g, target_licensed, max_unlicensed_ratio=unlicensed_ratio)
        
        print(f'Collected metadata for {len(rows)} repositories. Writing raw CSV...')
        
        # Write raw CSV
        if rows:
            keys = rows[0].keys()
            with open(output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            print(f'Wrote raw CSV to {output}')
    
    # --- Data cleaning & feature engineering ---
    cleaned = []
    for r in rows:
        cleaned.append({
            'full_name': r['full_name'],
            'stars': r['stars'] if isinstance(r['stars'], int) else (int(r['stars']) if r['stars'] else 0),
            'log_stars': math.log((r['stars'] if isinstance(r['stars'], int) else (int(r['stars']) if r['stars'] else 0)) + 1),
            'license_spdx': r.get('license_spdx') or None,
            'license_cat': classify_license(r.get('license_spdx')),
            'forks_count': r['forks_count'] or 0,
            'open_issues_count': r['open_issues_count'] or 0,
            'commit_count': r['commit_count'] or 0,
            'subscribers_count': r['subscribers_count'] or 0,
            'size': r['size'] or 0,
            'language': r.get('language') or 'Unknown',
            'owner_type': r.get('owner_type') or 'User',
            'age_days': days_since(r.get('created_at') or datetime.now().isoformat())
        })
    
    # Remove rows with extremely incomplete data
    complete = [r for r in cleaned if r['stars'] is not None and r['age_days'] >= 0]
    
    print(f'Filtered outliers. Remaining: {len(complete)}')
    
    # Find top K most common languages
    language_counts = {}
    for r in complete:
        lang = r['language']
        language_counts[lang] = language_counts.get(lang, 0) + 1
    
    # Sort by count and get top K (excluding 'Unknown')
    sorted_languages = sorted(
        [(lang, count) for lang, count in language_counts.items() if lang != 'Unknown'],
        key=lambda x: x[1],
        reverse=True
    )
    top_languages = [lang for lang, _ in sorted_languages[:language_topk]]
    
    print(f'\nTop {len(top_languages)} languages by repository count:')
    for i, (lang, count) in enumerate(sorted_languages[:language_topk], 1):
        print(f'  {i}. {lang:20s}: {count:4d} repos')
    
    # Collect 'other' licenses for investigation
    other_licenses = {}
    for r in complete:
        if r['license_cat'] == 'other':
            spdx = r['license_spdx'] or 'UNKNOWN'
            if spdx not in other_licenses:
                other_licenses[spdx] = {'count': 0, 'examples': []}
            other_licenses[spdx]['count'] += 1
            if len(other_licenses[spdx]['examples']) < 3:
                other_licenses[spdx]['examples'].append(r['full_name'])
    
    # Sort by count
    sorted_other_licenses = sorted(other_licenses.items(), key=lambda x: x[1]['count'], reverse=True)
    
    if sorted_other_licenses:
        print(f'\n{"OTHER LICENSE TYPES FOUND":-^80}')
        print(f'Found {len(sorted_other_licenses)} different license types categorized as "other"')
        print(f'Total repositories with "other" licenses: {sum(v["count"] for v in other_licenses.values())}')
        print('\nBreakdown by SPDX identifier:')
        for spdx, data in sorted_other_licenses:
            print(f'\n  {spdx}:')
            print(f'    Count: {data["count"]} repositories')
            print(f'    Examples: {", ".join(data["examples"][:3])}')
        
        # Save to separate file
        other_licenses_output = output.replace('.csv', '_other_licenses.json')
        with open(other_licenses_output, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_other_licenses': len(sorted_other_licenses),
                    'total_repositories': sum(v['count'] for v in other_licenses.values())
                },
                'licenses': [
                    {
                        'spdx_id': spdx,
                        'count': data['count'],
                        'example_repositories': data['examples']
                    }
                    for spdx, data in sorted_other_licenses
                ]
            }, f, indent=2)
        print(f'\nDetailed "other" licenses saved to: {other_licenses_output}')
    
    # Display license category distribution
    license_distribution = {}
    for r in complete:
        cat = r['license_cat']
        license_distribution[cat] = license_distribution.get(cat, 0) + 1
    
    print(f'\n{"LICENSE CATEGORY DISTRIBUTION":-^80}')
    for cat in ['permissive', 'weak_copyleft', 'strong_copyleft', 'other', 'no_license']:
        count = license_distribution.get(cat, 0)
        pct = (count / len(complete) * 100) if complete else 0
        print(f'  {cat.replace("_", " ").title():20s}: {count:4d} repos ({pct:5.1f}%)')
    
    # Create dummy variables for license categories and top languages
    # Note: We exclude one category from each group to avoid the dummy variable trap
    # Baseline categories: no_license (for licenses), first language (for languages)
    baseline_language = top_languages[0] if top_languages else None
    
    print(f'\nBaseline categories (excluded from model to avoid dummy variable trap):')
    print(f'  License baseline: no_license')
    if baseline_language:
        print(f'  Language baseline: {baseline_language}')
    
    enhanced_filtered = []
    for r in complete:
        record = {
            **r,
            'is_permissive': 1 if r['license_cat'] == 'permissive' else 0,
            'is_weak_copyleft': 1 if r['license_cat'] == 'weak_copyleft' else 0,
            'is_strong_copyleft': 1 if r['license_cat'] == 'strong_copyleft' else 0,
            'is_other_license': 1 if r['license_cat'] == 'other' else 0,
            # is_no_license excluded - serves as baseline
            'is_org': 1 if r['owner_type'] == 'Organization' else 0
        }
        
        # Add language dummy variables (excluding baseline)
        for lang in top_languages:
            if lang == baseline_language:
                continue  # Skip baseline language
            # Create valid column names by replacing special characters
            col_name = f'lang_{lang.replace("+", "plus").replace("#", "sharp").replace(" ", "_")}'
            record[col_name] = 1 if r['language'] == lang else 0
        
        # Add lang_other for languages not in top K (and not Unknown)
        is_other_lang = r['language'] not in top_languages and r['language'] != 'Unknown'
        record['lang_other'] = 1 if is_other_lang else 0
        
        # Add lang_unknown for repositories with unknown/null language
        record['lang_unknown'] = 1 if r['language'] == 'Unknown' else 0
        
        enhanced_filtered.append(record)
    
    # Output the results to a json file
    output_json = output.replace('.csv', '_cleaned.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(enhanced_filtered, f, indent=2)
    print(f'Wrote cleaned data to {output_json}')
    
    # --- Linear Regression Analysis ---
    print('\n' + '='*80)
    print('PERFORMING LINEAR REGRESSION ANALYSIS')
    print('='*80)
    
    if len(enhanced_filtered) < 10:
        print('Not enough data for regression analysis (need at least 10 samples)')
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(enhanced_filtered)
    
    # Define independent variables (features) and dependent variable (target)
    # Note: is_no_license excluded to avoid dummy variable trap (serves as baseline)
    # Note: watchers_count excluded as it mirrors star count; subscribers_count is actual watchers
    base_feature_columns = [
        'forks_count', 'open_issues_count', 
        'commit_count', 'subscribers_count', 'size', 'age_days',
        'is_permissive', 'is_weak_copyleft', 'is_strong_copyleft', 
        'is_other_license', 'is_org'
    ]
    
    # Add language dummy variables to features
    language_columns = [col for col in df.columns if col.startswith('lang_')]
    feature_columns = base_feature_columns + language_columns
    
    print(f'Including {len(language_columns)} language dummy variables in the model')
    
    # Remove rows with missing values in key columns
    df_clean = df.dropna(subset=['log_stars'] + feature_columns)
    
    if len(df_clean) < 10:
        print('Not enough complete data for regression analysis')
        return
    
    X = df_clean[feature_columns]
    y = df_clean['log_stars']
    
    print(f'\nRegression dataset: {len(df_clean)} repositories')
    print(f'Target variable: log(stars + 1)')
    print(f'Independent variables: {len(feature_columns)}')
    
    # Standardize features for better interpretation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f'\n{"Model Performance":-^80}')
    print(f'R² Score: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'Mean log(stars): {y.mean():.4f}')
    
    # Create results DataFrame
    coefficients_df = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f'\n{"Regression Coefficients (Standardized)":-^80}')
    print('(Larger absolute values = stronger effect on popularity)\n')
    print(coefficients_df.to_string(index=False))
    
    # Analyze license effects specifically
    print(f'\n{"LICENSE TYPE EFFECTS ON POPULARITY":-^80}')
    license_effects = coefficients_df[coefficients_df['Feature'].str.startswith('is_')]
    license_effects = license_effects[license_effects['Feature'].str.contains('license|permissive|copyleft')]
    
    print('\nRelative impact of license types (compared to no_license baseline):')
    print('Note: Positive = more popular than repos with no license')
    print('      Negative = less popular than repos with no license\n')
    for _, row in license_effects.iterrows():
        license_type = row['Feature'].replace('is_', '').replace('_', ' ').title()
        coef = row['Coefficient']
        direction = 'POSITIVE' if coef > 0 else 'NEGATIVE'
        print(f'  {license_type:25s}: {coef:+.4f} ({direction} effect)')
    
    # Summary statistics by license category
    print(f'\n{"Average Stars by License Category":-^80}')
    license_stats = df_clean.groupby('license_cat').agg({
        'stars': ['count', 'mean', 'median'],
        'log_stars': 'mean'
    }).round(2)
    license_stats.columns = ['Count', 'Mean Stars', 'Median Stars', 'Mean log(stars)']
    print(license_stats.to_string())
    
    # Interpretation
    print(f'\n{"KEY FINDINGS":-^80}')
    
    # Find the license with strongest positive effect
    license_coefs = {
        'permissive': coefficients_df[coefficients_df['Feature'] == 'is_permissive']['Coefficient'].values[0],
        'weak_copyleft': coefficients_df[coefficients_df['Feature'] == 'is_weak_copyleft']['Coefficient'].values[0],
        'strong_copyleft': coefficients_df[coefficients_df['Feature'] == 'is_strong_copyleft']['Coefficient'].values[0],
        'other_license': coefficients_df[coefficients_df['Feature'] == 'is_other_license']['Coefficient'].values[0],
        'no_license': 0.0  # Baseline category (excluded from model)
    }
    
    sorted_licenses = sorted(license_coefs.items(), key=lambda x: x[1], reverse=True)
    
    print(f'\n1. License ranking by positive effect on popularity:')
    for i, (lic, coef) in enumerate(sorted_licenses, 1):
        print(f'   {i}. {lic.replace("_", " ").title():20s} (coefficient: {coef:+.8f})')
    
    print(f'\n2. Most influential non-license factors:')
    non_license_features = coefficients_df[~coefficients_df['Feature'].str.startswith('is_')]
    top_factors = non_license_features.head(5)
    for _, row in top_factors.iterrows():
        feature_display = row['Feature'].replace('lang_', '').replace('plus', '+').replace('sharp', '#').replace('_', ' ')
        print(f'   - {feature_display:25s}: {row["Coefficient"]:+.4f}')
    
    # Show language effects specifically
    language_effects = coefficients_df[coefficients_df['Feature'].str.startswith('lang_')].sort_values('Coefficient', ascending=False)
    if len(language_effects) > 0:
        print(f'\n4. Programming language effects on popularity (top 5):')
        for _, row in language_effects.head(5).iterrows():
            lang_name = row['Feature'].replace('lang_', '').replace('plus', '+').replace('sharp', '#').replace('_', ' ')
            coef = row['Coefficient']
            direction = 'POSITIVE' if coef > 0 else 'NEGATIVE'
            print(f'   - {lang_name:20s}: {coef:+.4f} ({direction} effect)')
    
    print(f'\n3. Model explanation:')
    print(f'   - The model explains {r2*100:.1f}% of variance in repository popularity')
    print(f'   - All coefficients are standardized (comparable scale)')
    print(f'   - Positive coefficients = associated with MORE stars')
    print(f'   - Negative coefficients = associated with FEWER stars')
    print(f'   - License coefficients are relative to no_license baseline (coefficient = 0.0)')
    if baseline_language:
        print(f'   - Language coefficients are relative to {baseline_language} baseline')
    
    # --- Generate Visualizations for Thesis ---
    print('\n' + '='*80)
    print('GENERATING VISUALIZATIONS')
    print('='*80)
    
    # Set style for publication-quality figures
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig_dir = output.replace('.csv', '_figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. License Category Distribution (Bar Chart)
    print('Creating license distribution chart...')
    fig, ax = plt.subplots(figsize=(10, 6))
    license_order = ['permissive', 'weak_copyleft', 'strong_copyleft', 'other', 'no_license']
    license_labels = ['Permissive', 'Weak Copyleft', 'Strong Copyleft', 'Other', 'No License']
    license_counts = [license_distribution.get(cat, 0) for cat in license_order]
    colors = sns.color_palette("husl", len(license_order))
    
    bars = ax.bar(license_labels, license_counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('License Category', fontsize=12)
    ax.set_ylabel('Number of Repositories', fontsize=12)
    ax.set_title('Distribution of License Categories in Sample', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, count in zip(bars, license_counts):
        height = bar.get_height()
        ax.annotate(f'{count}\n({count/len(complete)*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '1_license_distribution.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, '1_license_distribution.pdf'), bbox_inches='tight')
    plt.close()
    
    # 2. Stars Distribution by License Type (Box Plot)
    print('Creating stars by license box plot...')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use log_stars for better visualization (stars are heavily skewed)
    box_data = [df_clean[df_clean['license_cat'] == cat]['log_stars'].values for cat in license_order]
    
    bp = ax.boxplot(box_data, labels=license_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('License Category', fontsize=12)
    ax.set_ylabel('log(Stars + 1)', fontsize=12)
    ax.set_title('Repository Popularity Distribution by License Type', fontsize=14, fontweight='bold')
    
    # Add sample sizes
    for i, cat in enumerate(license_order):
        n = len(df_clean[df_clean['license_cat'] == cat])
        ax.annotate(f'n={n}', xy=(i+1, ax.get_ylim()[0]), 
                    xytext=(0, -20), textcoords="offset points",
                    ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '2_stars_by_license_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, '2_stars_by_license_boxplot.pdf'), bbox_inches='tight')
    plt.close()
    
    # 3. Regression Coefficients (Horizontal Bar Chart) - All factors
    print('Creating regression coefficients chart...')
    
    # Show all factors, adjust figure height based on number of features
    all_coefs = coefficients_df.copy()
    all_coefs = all_coefs.sort_values('Coefficient')
    num_features = len(all_coefs)
    fig_height = max(8, num_features * 0.4)  # Scale height with number of features
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Format feature names for display
    display_names = []
    for feat in all_coefs['Feature']:
        name = feat.replace('is_', '').replace('lang_', 'Lang: ')
        name = name.replace('plus', '+').replace('sharp', '#').replace('_', ' ')
        name = name.replace('subscribers count', 'watcher count')  # subscribers_count is actually watchers
        name = name.title()
        display_names.append(name)
    
    colors_coef = ['#e74c3c' if c < 0 else '#2ecc71' for c in all_coefs['Coefficient']]
    
    bars = ax.barh(display_names, all_coefs['Coefficient'], color=colors_coef, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Standardized Coefficient', fontsize=12)
    ax.set_title('All Factors Influencing Repository Popularity\n(Multiple Linear Regression)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '3_regression_coefficients.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, '3_regression_coefficients.pdf'), bbox_inches='tight')
    plt.close()
    
    # 4. License Effect Comparison (Bar Chart) - License coefficients only
    print('Creating license effects comparison chart...')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    license_names_display = ['Permissive', 'Weak Copyleft', 'Strong Copyleft', 'Other', 'No License\n(Baseline)']
    license_coef_values = [license_coefs[k] for k in ['permissive', 'weak_copyleft', 'strong_copyleft', 'other_license', 'no_license']]
    
    colors_lic = ['#2ecc71' if c > 0 else '#e74c3c' if c < 0 else '#95a5a6' for c in license_coef_values]
    
    bars = ax.bar(license_names_display, license_coef_values, color=colors_lic, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('License Category', fontsize=12)
    ax.set_ylabel('Effect on log(Stars + 1) vs No License Baseline', fontsize=12)
    ax.set_title('License Type Effect on Repository Popularity\n(Controlling for Other Factors)', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, license_coef_values):
        height = bar.get_height()
        ax.annotate(f'{val:+.6f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '4_license_effects.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, '4_license_effects.pdf'), bbox_inches='tight')
    plt.close()
    
    # 5. Top Programming Languages Distribution (Bar Chart)
    print('Creating programming languages distribution chart...')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Count languages from the complete dataset
    lang_counts = {}
    for r in complete:
        lang = r.get('language', 'Unknown')
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    # Sort and get top K (include Unknown at the end if present)
    sorted_langs = sorted(
        [(lang, count) for lang, count in lang_counts.items() if lang != 'Unknown'],
        key=lambda x: x[1],
        reverse=True
    )[:language_topk]
    
    # Add "Other" category for remaining languages
    top_lang_names = [lang for lang, _ in sorted_langs]
    other_count = sum(count for lang, count in lang_counts.items() if lang not in top_lang_names and lang != 'Unknown')
    unknown_count = lang_counts.get('Unknown', 0)
    
    # Prepare data for chart
    lang_names = [lang for lang, _ in sorted_langs]
    lang_values = [count for _, count in sorted_langs]
    
    if other_count > 0:
        lang_names.append('Other')
        lang_values.append(other_count)
    if unknown_count > 0:
        lang_names.append('Unknown')
        lang_values.append(unknown_count)
    
    total_repos = len(complete)
    lang_colors = sns.color_palette("husl", len(lang_names))
    
    bars = ax.bar(lang_names, lang_values, color=lang_colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Programming Language', fontsize=12)
    ax.set_ylabel('Number of Repositories', fontsize=12)
    ax.set_title(f'Top {language_topk} Programming Languages in Sample', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, lang_values):
        height = bar.get_height()
        pct = (count / total_repos * 100)
        ax.annotate(f'{count}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '5_language_distribution.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, '5_language_distribution.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f'\nAll visualizations saved to: {fig_dir}/')
    print(f'  - PNG files (300 DPI) for presentations')
    print(f'  - PDF files for LaTeX/thesis inclusion')
    
    # Save regression results
    regression_output = output.replace('.csv', '_regression_results.txt')
    with open(regression_output, 'w') as f:
        f.write('LINEAR REGRESSION ANALYSIS: LICENSE EFFECT ON REPOSITORY POPULARITY\n')
        f.write('='*80 + '\n\n')
        f.write('BASELINE CATEGORIES (Dummy Variable Trap Avoidance):\n')
        f.write('  License baseline: no_license (coefficient = 0.0)\n')
        if baseline_language:
            f.write(f'  Language baseline: {baseline_language} (coefficient = 0.0)\n')
        f.write('  All coefficients are relative to these baselines\n\n')
        f.write(f'Model Performance:\n')
        f.write(f'  R² Score: {r2:.4f}\n')
        f.write(f'  RMSE: {rmse:.4f}\n')
        f.write(f'  Total Features: {len(feature_columns)}\n')
        f.write(f'  Language Features: {len(language_columns)}\n\n')
        f.write('Regression Coefficients:\n')
        f.write(coefficients_df.to_string(index=False))
        f.write('\n\nLicense Type Effects (relative to no_license baseline):\n')
        for lic, coef in sorted_licenses:
            f.write(f'  {lic.replace("_", " ").title():20s}: {coef:+.4f}\n')
        f.write('\n\nTop Programming Language Effects')
        if baseline_language:
            f.write(f' (relative to {baseline_language} baseline)')
        f.write(':\n')
        for _, row in language_effects.head(10).iterrows():
            lang_name = row['Feature'].replace('lang_', '').replace('plus', '+').replace('sharp', '#').replace('_', ' ')
            f.write(f'  {lang_name:20s}: {row["Coefficient"]:+.4f}\n')
    
    print(f'\nRegression results saved to: {regression_output}')
    print('='*80)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(f'Fatal error: {err}')
        import traceback
        traceback.print_exc()
        exit(1)
