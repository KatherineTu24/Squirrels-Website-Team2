#!/usr/bin/env python3
"""TTL (Turtle) parser for squirrel data"""

from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS
import json
from typing import Dict, List, Any
import re

# Define the namespace
BASE = Namespace("https://example.org/squirrel#")

def parse_ttl_file(ttl_file_path: str) -> Dict[str, Any]:
    """
    Parse TTL file and extract squirrel sighting data
    
    Returns:
        Dictionary with sightings, parks, areas, and fur colors
    """
    print(f"Loading TTL file: {ttl_file_path}")
    g = Graph()
    g.parse(ttl_file_path, format="turtle")
    print(f"Loaded {len(g)} triples from TTL file")
    
    sightings = []
    parks = {}
    areas = {}
    fur_colors = {}
    
    # Extract sightings
    for sighting_uri in g.subjects(predicate=RDF.type, object=BASE.Sighting):
        sighting_data = {
            'id': str(sighting_uri).split('#')[-1],  # Extract ID from URI
            'park_id': None,
            'area_id': None,
            'fur_color': None,
            'count': 1,
            'date': None,
            'season': None,
            'timestamp': None,
            'perch_type': None,
            'latitude': None,
            'longitude': None,
            'time_of_day': None,
            'weather_bucket': None,
            'activities': []
        }
        
        # Extract properties
        for pred, obj in g.predicate_objects(subject=sighting_uri):
            pred_name = str(pred).split('#')[-1]
            
            if pred_name == 'sightingId':
                sighting_data['id'] = str(obj)
            elif pred_name == 'inPark':
                park_id = str(obj).split('#')[-1]
                sighting_data['park_id'] = park_id
            elif pred_name == 'inArea':
                area_id = str(obj).split('#')[-1]
                sighting_data['area_id'] = area_id
            elif pred_name == 'hasFurColor':
                color_id = str(obj).split('#')[-1]
                sighting_data['fur_color'] = color_id.replace('fur_', '')
            elif pred_name == 'hasCount':
                sighting_data['count'] = int(str(obj))
            elif pred_name == 'observedOn':
                sighting_data['date'] = str(obj)
            elif pred_name == 'season':
                sighting_data['season'] = str(obj)
            elif pred_name == 'timestamp':
                sighting_data['timestamp'] = str(obj)
            elif pred_name == 'hasPerch':
                perch_id = str(obj).split('#')[-1]
                sighting_data['perch_type'] = perch_id.replace('perch_', '').replace('_', ' ')
            elif pred_name == 'hasLatitude':
                sighting_data['latitude'] = float(str(obj))
            elif pred_name == 'hasLongitude':
                sighting_data['longitude'] = float(str(obj))
            elif pred_name == 'latitude':
                sighting_data['latitude'] = float(str(obj))
            elif pred_name == 'longitude':
                sighting_data['longitude'] = float(str(obj))
            elif pred_name == 'hasTimeOfDay':
                sighting_data['time_of_day'] = str(obj)
            elif pred_name == 'hasWeatherBucket':
                sighting_data['weather_bucket'] = str(obj)
            elif pred_name == 'hasActivity':
                # Extract activity label from activity class
                activity_uri = str(obj)
                activity_label = None
                
                # Get activity label from the activity class
                for label in g.objects(subject=obj, predicate=RDFS.label):
                    activity_label = str(label)
                    break
                
                if activity_label:
                    sighting_data['activities'].append(activity_label)
                else:
                    # Fallback to URI name if no label found
                    activity_name = activity_uri.split('#')[-1].replace('activity_', '').replace('_', ' ')
                    sighting_data['activities'].append(activity_name)
        
        # If no activity specified, use default
        if not sighting_data['activities']:
            sighting_data['activities'] = ['Foraging']
        
        sightings.append(sighting_data)
    
    print(f"Extracted {len(sightings)} sightings")
    
    # Extract park information
    for park_uri in g.subjects(predicate=RDF.type, object=BASE.Park):
        park_id = str(park_uri).split('#')[-1]
        park_name = None
        
        # Get park label
        for label in g.objects(subject=park_uri, predicate=RDFS.label):
            park_name = str(label)
            break
        
        parks[park_id] = {
            'id': park_id,
            'name': park_name or f"Park {park_id}"
        }
    
    # Extract area information
    for area_uri in g.subjects(predicate=RDF.type, object=BASE.Area):
        area_id = str(area_uri).split('#')[-1]
        area_name = None
        
        # Get area label
        for label in g.objects(subject=area_uri, predicate=RDFS.label):
            area_name = str(label)
            break
        
        areas[area_id] = {
            'id': area_id,
            'name': area_name or f"Area {area_id}"
        }
    
    # Extract fur color information
    for color_uri in g.subjects(predicate=RDF.type, object=BASE.FurColor):
        color_id = str(color_uri).split('#')[-1]
        color_name = None
        
        # Get color label
        for label in g.objects(subject=color_uri, predicate=RDFS.label):
            color_name = str(label)
            break
        
        fur_colors[color_id] = {
            'id': color_id,
            'name': color_name or color_id.replace('fur_', '')
        }
    
    return {
        'sightings': sightings,
        'parks': parks,
        'areas': areas,
        'fur_colors': fur_colors
    }

def convert_ttl_to_csv_format(ttl_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert TTL data to CSV-like format for ML pipeline compatibility
    """
    csv_data = []
    
    for sighting in ttl_data['sightings']:
        park_info = ttl_data['parks'].get(sighting['park_id'], {})
        area_info = ttl_data['areas'].get(sighting['area_id'], {})
        
        # Handle multiple activities - join them with comma
        activities_str = ', '.join(sighting.get('activities', ['Foraging']))
        
        # Convert to CSV-like format
        csv_row = {
            'area_name': area_info.get('name', ''),
            'area_id': sighting['area_id'] or '',
            'park_name': park_info.get('name', ''),
            'park_id': sighting['park_id'] or '',
            'fur_color': sighting['fur_color'] or '',
            'location': sighting.get('perch_type', 'Ground Plane'),
            'activities': activities_str,
            'lat': sighting.get('latitude', '40.85941'),
            'lon': sighting.get('longitude', '-73.933936'),
            'date': sighting['date'] or '',
            'number_of_squirrels': sighting['count'],
            'time_of_day': sighting.get('time_of_day', 'afternoon'),
            'weather_bucket': sighting.get('weather_bucket', 'clear')
        }
        
        csv_data.append(csv_row)
    
    return csv_data

def get_statistics_from_ttl(ttl_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistics from TTL data
    """
    sightings = ttl_data['sightings']
    
    # Total observations
    total_observations = len(sightings)
    
    # Color distribution
    color_counts = {}
    for sighting in sightings:
        color = sighting['fur_color'] or 'unknown'
        color_counts[color] = color_counts.get(color, 0) + 1
    
    # Park distribution
    park_counts = {}
    for sighting in sightings:
        park_id = sighting['park_id'] or 'unknown'
        park_counts[park_id] = park_counts.get(park_id, 0) + 1
    
    # Season distribution
    season_counts = {}
    for sighting in sightings:
        season = sighting['season'] or 'unknown'
        season_counts[season] = season_counts.get(season, 0) + 1
    
    return {
        'total_observations': total_observations,
        'color_distribution': color_counts,
        'park_distribution': park_counts,
        'season_distribution': season_counts,
        'unique_parks': len(park_counts),
        'unique_areas': len(ttl_data['areas'])
    }

if __name__ == "__main__":
    # Test the parser
    ttl_data = parse_ttl_file("Ontology.tll")
    print(f"Parsed {len(ttl_data['sightings'])} sightings")
    print(f"Found {len(ttl_data['parks'])} parks")
    print(f"Found {len(ttl_data['areas'])} areas")
    
    # Test CSV conversion
    csv_data = convert_ttl_to_csv_format(ttl_data)
    print(f"Converted to {len(csv_data)} CSV-like rows")
    
    # Test statistics
    stats = get_statistics_from_ttl(ttl_data)
    print(f"Statistics: {stats}")
