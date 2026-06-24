"""
tools.py — Deterministic MLS search tools exposed to the LLM agent.

Three tools:
  search_listings      — filter + score + rank listings
  get_listing_detail   — full record by listing ID
  neighborhood_stats   — price benchmarks for a neighbourhood

The LLM decides WHEN and HOW to call these.
The functions just execute faithfully and return JSON strings.

Owner PII (name, phone) is never returned to the LLM.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_PATH = Path(__file__).parent.parent / "data" / "miami_mls_listings.csv"

_MLS_DF: Optional[pd.DataFrame] = None


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_mls() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Data quality fixes found during analysis:
    # MLS-100169: sqft = 50 for a $1.25M SFH — almost certainly a data entry error
    df.loc[df['listing_id'] == 'MLS-100169', 'sqft'] = None

    # Normalise feature strings to frozensets for O(1) membership tests
    df['feature_set'] = df['features'].fillna('').apply(
        lambda f: frozenset(x.strip().lower() for x in f.split(';') if x.strip())
    )

    # Lowercase index columns for case-insensitive matching
    df['neighborhood_lower'] = df['neighborhood'].str.lower().str.strip()
    df['city_lower'] = df['city'].str.lower().str.strip()
    df['property_type_lower'] = df['property_type'].str.lower().str.strip()

    return df


def get_df() -> pd.DataFrame:
    global _MLS_DF
    if _MLS_DF is None:
        _MLS_DF = _load_mls()
    return _MLS_DF


# ── Tool: search_listings ─────────────────────────────────────────────────────

def search_listings(
    neighborhoods: Optional[list[str]] = None,
    min_bedrooms: Optional[int] = None,
    max_bedrooms: Optional[int] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    property_types: Optional[list[str]] = None,
    required_features: Optional[list[str]] = None,
    preferred_features: Optional[list[str]] = None,
    max_results: int = 5,
    include_pending: bool = False,
) -> str:
    """Filter, score and rank MLS listings against buyer criteria."""
    df = get_df().copy()

    # ── Hard filters ──
    if not include_pending:
        df = df[df['listing_status'].isin(['Active', 'Active Under Contract'])]

    if neighborhoods:
        nbhd_lower = [n.lower().strip() for n in neighborhoods]
        df = df[df['neighborhood_lower'].isin(nbhd_lower)]

    if min_bedrooms is not None:
        df = df[df['bedrooms'].fillna(0) >= min_bedrooms]

    if max_bedrooms is not None:
        df = df[df['bedrooms'].fillna(99) <= max_bedrooms]

    if min_price is not None:
        df = df[df['price'] >= min_price]

    if max_price is not None:
        df = df[df['price'] <= max_price]

    if property_types:
        types_lower = [t.lower().strip() for t in property_types]
        df = df[df['property_type_lower'].isin(types_lower)]

    if required_features:
        for feat in [f.lower().strip() for f in required_features]:
            df = df[df['feature_set'].apply(lambda fs: feat in fs)]

    if df.empty:
        return json.dumps({
            "count": 0,
            "listings": [],
            "note": "No listings matched. Consider relaxing neighbourhood, price, or feature constraints and searching again."
        })

    # ── Scoring ──
    pref_lower = [f.lower().strip() for f in (preferred_features or [])]

    def score(row):
        s = 0
        for feat in pref_lower:
            if feat in row['feature_set']:
                s += 2
        if row['days_on_market'] < 30:
            s += 1
        if row['listing_status'] == 'Active':
            s += 1
        if max_price and row['price'] < max_price * 0.9:
            s += 1
        return s

    df['_score'] = df.apply(score, axis=1)
    df = df.sort_values('_score', ascending=False).head(max_results)

    # ── Format — strip owner PII ──
    results = []
    for _, row in df.iterrows():
        matched_pref = [f for f in pref_lower if f in row['feature_set']]
        results.append({
            "listing_id": row['listing_id'],
            "address": row['address'],
            "neighborhood": row['neighborhood'],
            "city": row['city'],
            "price": int(row['price']),
            "bedrooms": None if pd.isna(row['bedrooms']) else int(row['bedrooms']),
            "bathrooms": None if pd.isna(row['bathrooms']) else float(row['bathrooms']),
            "sqft": None if pd.isna(row['sqft']) else int(row['sqft']),
            "year_built": int(row['year_built']),
            "property_type": row['property_type'],
            "listing_status": row['listing_status'],
            "days_on_market": int(row['days_on_market']),
            "features": row['features'],
            "description": row['description'],
            "preferred_features_matched": matched_pref,
            "match_score": int(row['_score']),
        })

    return json.dumps({"count": len(results), "listings": results}, indent=2)


# ── Tool: get_listing_detail ──────────────────────────────────────────────────

def get_listing_detail(listing_id: str) -> str:
    """Return full MLS record for a specific listing_id. Strips owner PII."""
    df = get_df()
    row = df[df['listing_id'] == listing_id]

    if row.empty:
        return json.dumps({"error": f"Listing '{listing_id}' not found in MLS dataset."})

    r = row.iloc[0]

    notes = []
    if pd.isna(r.get('sqft')) or (not pd.isna(r.get('sqft')) and r['sqft'] < 100):
        notes.append("sqft data appears unreliable — verify with listing agent before presenting to buyer.")
    if r['price'] > 10_000_000:
        notes.append("Ultra-luxury tier — verify price is not a data entry error.")
    if pd.isna(r.get('bedrooms')):
        notes.append("Bedroom count missing from MLS record — confirm with listing agent.")

    return json.dumps({
        "listing_id": r['listing_id'],
        "address": r['address'],
        "neighborhood": r['neighborhood'],
        "city": r['city'],
        "price": int(r['price']),
        "bedrooms": None if pd.isna(r['bedrooms']) else int(r['bedrooms']),
        "bathrooms": None if pd.isna(r['bathrooms']) else float(r['bathrooms']),
        "sqft": None if pd.isna(r['sqft']) else int(r['sqft']),
        "year_built": int(r['year_built']),
        "property_type": r['property_type'],
        "listing_status": r['listing_status'],
        "days_on_market": int(r['days_on_market']),
        "description": r['description'],
        "features": r['features'],
        "data_quality_notes": notes,
    }, indent=2)


# ── Tool: neighborhood_stats ──────────────────────────────────────────────────

def neighborhood_stats(neighborhood: str) -> str:
    """
    Price statistics and inventory summary for a neighbourhood.
    Use when buyer budget seems misaligned with preferred area,
    or to find and compare alternative neighbourhoods.
    """
    df = get_df()
    nbhd_lower = neighborhood.lower().strip()
    subset = df[df['neighborhood_lower'] == nbhd_lower]

    if subset.empty:
        partial = df[df['neighborhood_lower'].str.contains(nbhd_lower, na=False)]
        if partial.empty:
            available = sorted(df['neighborhood'].unique().tolist())
            return json.dumps({
                "error": f"Neighbourhood '{neighborhood}' not found.",
                "available_neighborhoods": available
            })
        subset = partial

    active = subset[subset['listing_status'].isin(['Active', 'Active Under Contract'])]

    def stats_for(frame):
        if frame.empty:
            return None
        prices = frame['price'].dropna()
        return {
            "count": len(frame),
            "median_price": int(prices.median()),
            "min_price": int(prices.min()),
            "max_price": int(prices.max()),
            "avg_price": int(prices.mean()),
            "property_type_breakdown": frame['property_type'].value_counts().to_dict(),
            "avg_days_on_market": round(float(frame['days_on_market'].mean()), 1),
        }

    return json.dumps({
        "neighborhood": subset['neighborhood'].iloc[0],
        "active_listings": stats_for(active),
        "all_listings": stats_for(subset),
    }, indent=2)


# ── Tool definitions for Anthropic API ───────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_listings",
        "description": (
            "Search and rank MLS listings against buyer criteria. "
            "This is your primary discovery tool — call it multiple times with different "
            "parameters to adapt your strategy. If the buyer's preferred neighbourhood yields "
            "few results, broaden to adjacent areas. Always pass max_price from the buyer's budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "neighborhoods": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Neighbourhood names to filter by. Pass [] or omit to search all."
                },
                "min_bedrooms": {"type": "integer", "description": "Minimum bedrooms (inclusive)."},
                "max_bedrooms": {"type": "integer", "description": "Maximum bedrooms (inclusive)."},
                "min_price": {"type": "number"},
                "max_price": {"type": "number", "description": "Buyer's stated budget ceiling."},
                "property_types": {
                    "type": "array", "items": {"type": "string"},
                    "description": "e.g. ['condo','single family','townhouse','villa','multi-family']"
                },
                "required_features": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "Must-have features (lowercase). Known values: pool, gym, balcony, garage, "
                        "waterfront, ocean view, bay view, boat dock, pet friendly, home office, "
                        "tennis court, rooftop, concierge, doorman, gated community, smart home, "
                        "updated kitchen, modern kitchen, large lot, private beach access, central ac, "
                        "terrace, garden, hardwood floors, marble floors, high ceilings, walk-in closet."
                    )
                },
                "preferred_features": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Nice-to-have features — boost ranking score but do not filter."
                },
                "max_results": {"type": "integer", "description": "Max listings to return (default 5)."},
                "include_pending": {"type": "boolean", "description": "Include Pending listings (default false)."},
            },
            "required": [],
        },
    },
    {
        "name": "get_listing_detail",
        "description": (
            "Get the full MLS record for a specific listing by its listing_id. "
            "Use this after identifying promising matches to get complete details, "
            "or when a buyer mentions a specific address."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "listing_id": {
                    "type": "string",
                    "description": "e.g. 'MLS-100042'"
                }
            },
            "required": ["listing_id"],
        },
    },
    {
        "name": "neighborhood_stats",
        "description": (
            "Get price statistics and inventory for a neighbourhood. "
            "Use when a buyer's budget seems too low for their preferred area "
            "(check the median price), or to identify and compare alternative areas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "neighborhood": {
                    "type": "string",
                    "description": "e.g. 'Brickell', 'Coral Gables', 'Edgewater'"
                }
            },
            "required": ["neighborhood"],
        },
    },
]


def dispatch(tool_name: str, tool_input: dict) -> str:
    """Route an agent tool call to the correct function."""
    if tool_name == "search_listings":
        return search_listings(**tool_input)
    elif tool_name == "get_listing_detail":
        return get_listing_detail(**tool_input)
    elif tool_name == "neighborhood_stats":
        return neighborhood_stats(**tool_input)
    else:
        return json.dumps({"error": f"Unknown tool: '{tool_name}'"})
