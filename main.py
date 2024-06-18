import os
import urllib.request, json
from typing import List

import requests

KEY = os.getenv('NS_KEY')


def get_all_vehicles():
    url = "https://gateway.apiportal.ns.nl/virtual-train-api/api/v1/trein"

    hdr = {
        # Request headers
        'Cache-Control': 'no-cache',
        'Ocp-Apim-Subscription-Key': KEY,
    }

    req = urllib.request.Request(url, headers=hdr)

    req.get_method = lambda: 'GET'
    response = urllib.request.urlopen(req)

    return json.loads(response.read().decode('utf-8'))


def get_train_info_for_ids(ids: List[int]):
    url = "https://gateway.apiportal.ns.nl/virtual-train-api/api/v1/trein?ids=" + ",".join(map(str, ids))

    hdr = {
        # Request headers
        'Cache-Control': 'no-cache',
        'Ocp-Apim-Subscription-Key': KEY,
    }

    req = urllib.request.Request(url, headers=hdr)

    req.get_method = lambda: 'GET'
    response = urllib.request.urlopen(req)

    return json.loads(response.read().decode('utf-8'))


def download_img(url: str, name: str):
    response = requests.get(url)
    if response.status_code == 200:
        # Save the image to the specified path
        location = f"./out/{name.replace('/', '_')}"
        with open(location, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download {url}")


if __name__ == '__main__':
    # Get all vehicles
    vehicles = get_all_vehicles()
    # Collect a train id for each train type
    vehicle_types = dict()
    for trip_id, vehicle in vehicles.items():
        for deel in list(vehicle.values())[0]['treindelen']:
            v_type = deel['type']
            if v_type not in vehicle_types:
                vehicle_types[v_type] = trip_id
    # Get full information for those trains
    vehicles = get_train_info_for_ids(list(vehicle_types.values()))
    # Download all images
    visited = set()
    for vehicle in vehicles:
        for deel in vehicle['materieeldelen']:
            deel_type = deel['type']
            if deel_type not in visited:
                visited.add(deel_type)
                bakken = deel['bakken']
                try:
                    download_img(bakken[0]['afbeelding']['url'], f"{deel_type}.png")
                    download_img(bakken[1]['afbeelding']['url'], f"{deel_type}-bak.png")
                    download_img(bakken[-1]['afbeelding']['url'], f"{deel_type}'.png")
                except (IndexError, KeyError):
                    pass
