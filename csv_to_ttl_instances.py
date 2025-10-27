#!/usr/bin/env python3
"""
Convert CSV squirrel data to TTL instances and append to Ontology.tll
"""

import csv
import re
from datetime import datetime

def clean_identifier(text):
    """Convert text to valid TTL identifier"""
    # Replace spaces and special chars with underscores, remove quotes
    cleaned = re.sub(r'[^\w\-]', '_', text.strip().replace('"', ''))
    # Remove multiple underscores and ensure it starts with letter
    cleaned = re.sub(r'_+', '_', cleaned)
    if cleaned and cleaned[0].isdigit():
        cleaned = 'id_' + cleaned
    return cleaned.lower()

def parse_date(date_str):
    """Convert date string to ISO format"""
    try:
        # Handle MM/DD/YY format
        if '/' in date_str:
            date_obj = datetime.strptime(date_str, '%m/%d/%y')
            return date_obj.strftime('%Y-%m-%d')
    except:
        pass
    return date_str

def split_activities(activity_str):
    """Split comma-separated activities and clean them"""
    if not activity_str or activity_str.strip() == '':
        return []
    activities = [act.strip() for act in activity_str.split(',')]
    return [act for act in activities if act]

def convert_csv_to_ttl():
    """Convert CSV data to TTL instances"""
    
    # Read CSV data
    instances = []
    areas = set()
    parks = set()
    activities = set()
    fur_colors = set()
    perches = set()
    
    processed_count = 0
    filtered_count = 0
    
    with open('squirrel-park_merged_clean.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            processed_count += 1
            # Skip empty rows
            if not row.get('squirrel_id') or row['squirrel_id'].strip() == '':
                filtered_count += 1
                continue
                
            # Extract data (handle BOM in first column)
            area_name = row.get('\ufeffarea_name', row.get('area_name', '')).strip()
            park_name = row.get('park_name', '').strip()
            squirrel_id = row.get('squirrel_id', '').strip()
            fur_color = row.get('fur_color', '').strip()
            location = row.get('location', '').strip()
            activities_str = row.get('activities', '').strip()
            lat = row.get('lat', '').strip()
            lon = row.get('lon', '').strip()
            date_str = row.get('date', '').strip()
            squirrel_count = row.get('number_of_squirrels', '1').strip()
            
            # Skip if essential data is missing (location/perch is now optional)
            if not all([area_name, park_name, squirrel_id, fur_color, activities_str]):
                filtered_count += 1
                continue
            
            # Create identifiers
            area_id = clean_identifier(area_name)
            park_id = clean_identifier(park_name)
            sighting_id = clean_identifier(squirrel_id)
            fur_color_id = clean_identifier(fur_color)
            perch_id = clean_identifier(location)
            
            # Parse activities
            activity_list = split_activities(activities_str)
            activity_ids = [clean_identifier(act) for act in activity_list]
            
            # Collect unique values
            areas.add((area_id, area_name))
            parks.add((park_id, park_name))
            fur_colors.add((fur_color_id, fur_color))
            if location:  # Only collect perch if location is available
                perches.add((perch_id, location))
            for act in activity_list:
                activities.add((clean_identifier(act), act))
            
            # Create sighting instance
            sighting_ttl = f":sighting_{sighting_id} a :Sighting ;\n"
            sighting_ttl += f"    :sightingId \"{squirrel_id}\" ;\n"
            sighting_ttl += f"    :observedOn \"{parse_date(date_str)}\"^^xsd:date ;\n"
            sighting_ttl += f"    :hasCount {squirrel_count} ;\n"
            if lat and lon:
                sighting_ttl += f"    :latitude {lat} ;\n"
                sighting_ttl += f"    :longitude {lon} ;\n"
            sighting_ttl += f"    :inPark :park_{park_id} ;\n"
            sighting_ttl += f"    :inArea :area_{area_id} ;\n"
            sighting_ttl += f"    :hasFurColor :fur_{fur_color_id} ;\n"
            if location:  # Only add perch if location is available
                sighting_ttl += f"    :hasPerch :perch_{perch_id} ;\n"
            
            # Add activities
            for act_id in activity_ids:
                sighting_ttl += f"    :hasActivity :activity_{act_id} ;\n"
            
            # Remove last semicolon and add period
            sighting_ttl = sighting_ttl.rstrip(' ;\n') + " .\n\n"
            instances.append(sighting_ttl)
    
    
    # Generate TTL instances
    ttl_content = "\n### Area instances\n"
    for area_id, area_name in sorted(areas):
        ttl_content += f":area_{area_id} a :Area ;\n"
        ttl_content += f"    rdfs:label \"{area_name}\"@en .\n\n"
    
    # Create park-area mapping
    park_area_map = {}
    with open('squirrel-park_merged_clean.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            park_name = row.get('park_name', '').strip()
            area_name = row.get('\ufeffarea_name', row.get('area_name', '')).strip()
            if park_name and area_name:
                park_area_map[clean_identifier(park_name)] = clean_identifier(area_name)
    
    ttl_content += "### Park instances\n"
    for park_id, park_name in sorted(parks):
        area_id = park_area_map.get(park_id, 'unknown')
        ttl_content += f":park_{park_id} a :Park ;\n"
        ttl_content += f"    rdfs:label \"{park_name}\"@en ;\n"
        ttl_content += f"    :parkInArea :area_{area_id} .\n\n"
    
    ttl_content += "### Fur Color instances\n"
    for fur_id, fur_name in sorted(fur_colors):
        ttl_content += f":fur_{fur_id} a :FurColor ;\n"
        ttl_content += f"    rdfs:label \"{fur_name}\"@en .\n\n"
    
    ttl_content += "### Perch instances\n"
    for perch_id, perch_name in sorted(perches):
        ttl_content += f":perch_{perch_id} a :Perch ;\n"
        ttl_content += f"    rdfs:label \"{perch_name}\"@en .\n\n"
    
    ttl_content += "### Activity instances\n"
    for act_id, act_name in sorted(activities):
        ttl_content += f":activity_{act_id} a :Activity ;\n"
        ttl_content += f"    rdfs:label \"{act_name}\"@en .\n\n"
    
    ttl_content += "### Sighting instances\n"
    ttl_content += "".join(instances)
    
    print(f"Debug: Processed {processed_count} CSV rows")
    print(f"Debug: Filtered out {filtered_count} rows")
    print(f"Debug: Generated {len(instances)} sighting instances")
    
    return ttl_content

def main():
    """Main function to generate and append TTL instances"""
    print("Converting CSV to TTL instances...")
    
    # Generate TTL content
    ttl_instances = convert_csv_to_ttl()
    
    # Append to ontology file
    with open('Ontology.tll', 'a', encoding='utf-8') as f:
        f.write(ttl_instances)
    
    print(f"Successfully added TTL instances to Ontology.tll")
    print(f"Added instances for areas, parks, fur colors, perches, activities, and sightings")

if __name__ == "__main__":
    main()
