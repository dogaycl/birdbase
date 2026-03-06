import requests

def get_species_info(species_name: str) -> dict:
    """Retrieve detailed information about a bird species from Wikipedia."""
    # Clean the name (e.g., 'Black footed Albatross' -> 'Black-footed Albatross' or similar)
    search_query = species_name.title()
    
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{search_query}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "name": species_name,
                "scientific_name": "Scientific name usually in Wikipedia text",
                "habitat": "Refer to Wikipedia",
                "lifespan": "Refer to Wikipedia",
                "description": data.get("extract", "Description not found on Wikipedia.")
            }
        else:
            return {
                "name": species_name,
                "scientific_name": "Unknown",
                "habitat": "Unknown",
                "lifespan": "Unknown",
                "description": f"Detailed information for {species_name} is not available locally."
            }
    except Exception as e:
        return {"error": str(e), "name": species_name}

