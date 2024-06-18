import json
import os
import urllib.request
from typing import List

import cv2
import numpy as np
import requests
from PIL import Image

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


def make_transparent(image_path: str):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Apply mask to the alpha channel
    alpha_channel = np.ones_like(mask) * 255
    alpha_channel[mask == 255] = 0

    # Set alpha channel
    img_rgba[:, :, 3] = alpha_channel
    img_rgba[:, :, 2] = img_rgb[:, :, 2]
    img_rgba[:, :, 1] = img_rgb[:, :, 1]
    img_rgba[:, :, 0] = img_rgb[:, :, 0]

    # Save the image with transparency
    output_image = Image.fromarray(img_rgba)
    return output_image


def download_img(url: str, name: str):
    response = requests.get(url)
    if response.status_code == 200:
        # Save the image to the specified path
        location = f"./out/{name.replace('/', '_')}"
        with open(location, 'wb') as file:
            file.write(response.content)
        # Check for transparency
        img = Image.open(location)
        if img.mode == 'RGBA' and img.getchannel('A').getextrema() != (255, 255):
            pass
        else:
            img = make_transparent(location)
            img.save(location, 'png')
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
