import json
import os
import urllib.request
from enum import Enum
from typing import List, Optional
from urllib.error import HTTPError

import cv2
import numpy as np
import requests
from PIL import Image

KEY = os.getenv('NS_KEY')
if not KEY:
    print("NS_KEY environment variable not set")
    exit(1)


def get_all_vehicles():
    url = "https://gateway.apiportal.ns.nl/virtual-train-api/api/v1/trein"

    hdr = {
        # Request headers
        'Cache-Control': 'no-cache',
        'Ocp-Apim-Subscription-Key': KEY,
    }

    req = urllib.request.Request(url, headers=hdr)

    req.get_method = lambda: 'GET'
    try:
        response = urllib.request.urlopen(req)
    except HTTPError as e:
        print(f"Error {e.code} getting all vehicles: {e.reason}")
        print(f"{e.read().decode()}")
        exit(2)

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
    try:
        response = urllib.request.urlopen(req)
    except HTTPError as e:
        print(f"Error {e.code} getting train info: {e.reason}")
        print(f"{e.read().decode()}")
        exit(2)


    return json.loads(response.read().decode('utf-8'))


def make_transparent(image_path: str) -> Image:
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


class Part(Enum):
    HEAD = 0
    MIDDLE = 1
    TAIL = 2


def crop_to_square(img: Image, part: Part) -> Image:
    width, height = img.size
    new_size = min(width, height)
    match part:
        case Part.HEAD:
            left = 0
            right = new_size
        case Part.MIDDLE:
            left = (width - new_size) / 2
            right = (width + new_size) / 2
        case Part.TAIL:
            left = width - new_size
            right = width
        case _:
            raise NotImplementedError()
    top = (height - new_size) / 2
    bottom = (height + new_size) / 2

    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped


def download_img(url: str, name: str, part: Optional[Part]):
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
        # Crop image
        if part is None:
            img.save(location[:-4] + "-full.png", 'png')
            old_img = img.copy()
            img = crop_to_square(old_img, Part.HEAD)
            img.save(location[:-4] + ".png", 'png')
            img = crop_to_square(old_img, Part.MIDDLE)
            img.save(location[:-4] + "-bak.png", 'png')
            img = crop_to_square(old_img, Part.TAIL)
            img.save(location[:-4] + "'.png", 'png')
        else:
            img = crop_to_square(img, part)
            img.save(location, 'png')
    else:
        print(f"Failed to download {url}")


if __name__ == '__main__':
    # Get all vehicles
    print("Getting all vehicles...")
    vehicles = get_all_vehicles()
    # Collect a train id for each train type
    vehicle_types = dict()
    for trip_id, vehicle in vehicles.items():
        for deel in list(vehicle.values())[0]['treindelen']:
            v_type = deel['type']
            if v_type not in vehicle_types:
                vehicle_types[v_type] = trip_id
    # Get full information for those trains
    print("Getting vehicle information...")
    vehicles = get_train_info_for_ids(list(vehicle_types.values()))
    # Download all images
    print("Downloading images...")
    visited = set()
    for vehicle in vehicles:
        for deel in vehicle['materieeldelen']:
            deel_type = deel['type']
            if deel_type not in visited:
                bakken = deel['bakken']
                try:
                    if len(bakken) == 0:
                        download_img(deel['afbeelding'], f"{deel_type}.png", None)
                    else:
                        download_img(bakken[0]['afbeelding']['url'], f"{deel_type}.png", Part.HEAD)
                        download_img(bakken[1]['afbeelding']['url'], f"{deel_type}-bak.png", Part.MIDDLE)
                        download_img(bakken[-1]['afbeelding']['url'], f"{deel_type}'.png", Part.TAIL)
                    visited.add(deel_type)
                except (IndexError, KeyError):
                    pass
