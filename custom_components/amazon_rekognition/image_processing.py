"""
Platform that will perform object detection.
"""
from collections import namedtuple, Counter
import io
import logging
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw, UnidentifiedImageError, ImageFont

import homeassistant.helpers.config_validation as cv
import homeassistant.util.dt as dt_util
import voluptuous as vol
from homeassistant.components.image_processing import (
    ATTR_CONFIDENCE,
    CONF_CONFIDENCE,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    DEFAULT_CONFIDENCE,
    DOMAIN,
    PLATFORM_SCHEMA,
    ImageProcessingEntity,
)
from homeassistant.core import split_entity_id
from homeassistant.util.pil import draw_box

from homeassistant.const import ATTR_ENTITY_ID, ATTR_NAME

_LOGGER = logging.getLogger(__name__)

CONF_REGION = "region_name"
CONF_ACCESS_KEY_ID = "aws_access_key_id"
CONF_SECRET_ACCESS_KEY = "aws_secret_access_key"
CONF_COLLECTION_ID = "collection_id"

DEFAULT_REGION = "us-east-1"
SUPPORTED_REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ca-central-1",
    "eu-west-1",
    "eu-central-1",
    "eu-west-2",
    "eu-west-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-northeast-2",
    "ap-northeast-1",
    "ap-south-1",
    "sa-east-1",
]

CONF_BOTO_RETRIES = "boto_retries"
CONF_SAVE_FILE_FORMAT = "save_file_format"
CONF_SAVE_FILE_FOLDER = "save_file_folder"
CONF_SAVE_TIMESTAMPED_FILE = "save_timestamped_file"
CONF_ALWAYS_SAVE_LATEST_FILE = "always_save_latest_file"
CONF_SHOW_BOXES = "show_boxes"
CONF_SCALE = "scale"
CONF_TARGET = "target"
CONF_TARGETS = "targets"
CONF_S3_BUCKET = "s3_bucket"

CONF_ROI_Y_MIN = "roi_y_min"
CONF_ROI_X_MIN = "roi_x_min"
CONF_ROI_Y_MAX = "roi_y_max"
CONF_ROI_X_MAX = "roi_x_max"

DATETIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
DEFAULT_BOTO_RETRIES = 5
PERSON = "person"
DEFAULT_TARGETS = [{CONF_TARGET: PERSON}]
DEFAULT_ROI_Y_MIN = 0.0
DEFAULT_ROI_Y_MAX = 1.0
DEFAULT_ROI_X_MIN = 0.0
DEFAULT_ROI_X_MAX = 1.0
DEFAULT_SCALE = 1.0
DEFAULT_ROI = (
    DEFAULT_ROI_Y_MIN,
    DEFAULT_ROI_X_MIN,
    DEFAULT_ROI_Y_MAX,
    DEFAULT_ROI_X_MAX,
)

EVENT_OBJECT_DETECTED = "rekognition.object_detected"
EVENT_LABEL_DETECTED = "rekognition.label_detected"

BOX = "box"
FILE = "file"
OBJECT = "object"
SAVED_FILE = "saved_file"
MIN_CONFIDENCE = 0.1
JPG = "jpg"
PNG = "png"

# rgb(red, green, blue)
RED = (255, 0, 0)  # For objects within the ROI
GREEN = (0, 255, 0)  # For ROI box
YELLOW = (255, 255, 0)  # For objects outside the ROI

REQUIREMENTS = ["boto3"]

TARGETS_SCHEMA = {
    vol.Required(CONF_TARGET): cv.string,
    vol.Optional(CONF_CONFIDENCE): vol.All(
        vol.Coerce(float), vol.Range(min=10, max=100)
    ),
}

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_REGION, default=DEFAULT_REGION): vol.In(SUPPORTED_REGIONS),
        vol.Required(CONF_ACCESS_KEY_ID): cv.string,
        vol.Required(CONF_SECRET_ACCESS_KEY): cv.string,
        vol.Optional(CONF_TARGETS, default=DEFAULT_TARGETS): vol.All(
            cv.ensure_list, [vol.Schema(TARGETS_SCHEMA)]
        ),
        vol.Optional(CONF_ROI_Y_MIN, default=DEFAULT_ROI_Y_MIN): cv.small_float,
        vol.Optional(CONF_ROI_X_MIN, default=DEFAULT_ROI_X_MIN): cv.small_float,
        vol.Optional(CONF_ROI_Y_MAX, default=DEFAULT_ROI_Y_MAX): cv.small_float,
        vol.Optional(CONF_ROI_X_MAX, default=DEFAULT_ROI_X_MAX): cv.small_float,
        vol.Optional(CONF_SCALE, default=DEFAULT_SCALE): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=1)
        ),
        vol.Optional(CONF_SAVE_FILE_FOLDER): cv.isdir,
        vol.Optional(CONF_SAVE_FILE_FORMAT, default=JPG): vol.In([JPG, PNG]),
        vol.Optional(CONF_SAVE_TIMESTAMPED_FILE, default=False): cv.boolean,
        vol.Optional(CONF_ALWAYS_SAVE_LATEST_FILE, default=False): cv.boolean,
        vol.Optional(CONF_S3_BUCKET): cv.string,
        vol.Optional(CONF_SHOW_BOXES, default=True): cv.boolean,
        vol.Optional(CONF_BOTO_RETRIES, default=DEFAULT_BOTO_RETRIES): vol.All(
            vol.Coerce(int), vol.Range(min=0)
        ),
        # ✅ Aggiunta di collection_id
        vol.Optional(CONF_COLLECTION_ID): cv.string,
    }
)


Box = namedtuple("Box", "y_min x_min y_max x_max")
Point = namedtuple("Point", "y x")


def point_in_box(box: Box, point: Point) -> bool:
    """Return true if point lies in box"""
    if (box.x_min <= point.x <= box.x_max) and (box.y_min <= point.y <= box.y_max):
        return True
    return False


def object_in_roi(roi: dict, centroid: dict) -> bool:
    """Convenience to convert dicts to the Point and Box."""
    target_center_point = Point(centroid["y"], centroid["x"])
    roi_box = Box(roi["y_min"], roi["x_min"], roi["y_max"], roi["x_max"])
    return point_in_box(roi_box, target_center_point)


def get_objects(response: str) -> dict:
    _LOGGER.debug("get_objects received response: %s", response)
    _LOGGER.debug("modifica apporartata.")
    """Parse the data, returning detected objects only."""
    objects = []
    labels = []
    decimal_places = 3

    for label in response["Labels"]:
        if len(label["Instances"]) > 0:
            for instance in label["Instances"]:
                # Extract and format instance data
                box = instance["BoundingBox"]
                # Get bounding box
                x_min, y_min, width, height = (
                    box["Left"],
                    box["Top"],
                    box["Width"],
                    box["Height"],
                )
                x_max, y_max = x_min + width, y_min + height

                bounding_box = {
                    "x_min": round(x_min, decimal_places),
                    "y_min": round(y_min, decimal_places),
                    "x_max": round(x_max, decimal_places),
                    "y_max": round(y_max, decimal_places),
                    "width": round(box["Width"], decimal_places),
                    "height": round(box["Height"], decimal_places),
                }

                # Get box area (% of frame)
                box_area = width * height * 100

                # Get box centroid
                centroid_x, centroid_y = (x_min + width / 2), (y_min + height / 2)
                centroid = {
                    "x": round(centroid_x, decimal_places),
                    "y": round(centroid_y, decimal_places),
                }

                objects.append(
                    {
                        "name": label["Name"].lower(),
                        "confidence": round(instance["Confidence"], decimal_places),
                        "bounding_box": bounding_box,
                        "box_area": round(box_area, decimal_places),
                        "centroid": centroid,
                    }
                )
        else:
            label_info = {
                "name": label["Name"].lower(),
                "confidence": round(label["Confidence"], decimal_places),
            }
            labels.append(label_info)
    return objects, labels


def get_valid_filename(name: str) -> str:
    return re.sub(r"(?u)[^-\w.]", "", str(name).strip().replace(" ", "_"))


def setup_platform(hass, config, add_devices, discovery_info=None):
    """Set up ObjectDetection."""
    collection_id = config.get(CONF_COLLECTION_ID)
    import boto3

    _LOGGER.debug("boto_retries setting is %s", config[CONF_BOTO_RETRIES])

    aws_config = {
        CONF_REGION: config[CONF_REGION],
        CONF_ACCESS_KEY_ID: config[CONF_ACCESS_KEY_ID],
        CONF_SECRET_ACCESS_KEY: config[CONF_SECRET_ACCESS_KEY],
    }

    retries = 0
    success = False
    rekognition_client = None

    while retries <= config[CONF_BOTO_RETRIES]:
        try:
            rekognition_client = boto3.client("rekognition", **aws_config)
            success = True
            break
        except Exception as e:
            _LOGGER.warning("boto3 client failed, retry {}/{}: {}".format(retries, config[CONF_BOTO_RETRIES], e))
            retries += 1
            time.sleep(1)

    if not success:
        raise Exception(
            "Failed to create boto3 client. Increase boto_retries setting. Retries: {}".format(retries)
        )

    s3_client = boto3.client("s3", **aws_config) if CONF_S3_BUCKET in config else None

    save_file_folder = config.get(CONF_SAVE_FILE_FOLDER)
    if save_file_folder:
        save_file_folder = Path(save_file_folder)

    entities = []
    for camera in config[CONF_SOURCE]:
        entities.append(
            ObjectDetection(
                rekognition_client=rekognition_client,
                s3_client=s3_client,
                region=config.get(CONF_REGION),
                targets=config.get(CONF_TARGETS),
                confidence=config.get(CONF_CONFIDENCE),
                roi_y_min=config[CONF_ROI_Y_MIN],
                roi_x_min=config[CONF_ROI_X_MIN],
                roi_y_max=config[CONF_ROI_Y_MAX],
                roi_x_max=config[CONF_ROI_X_MAX],
                scale=config[CONF_SCALE],
                show_boxes=config[CONF_SHOW_BOXES],
                save_file_format=config[CONF_SAVE_FILE_FORMAT],
                save_file_folder=save_file_folder,
                save_timestamped_file=config.get(CONF_SAVE_TIMESTAMPED_FILE),  # Corretto
                always_save_latest_file=config.get(CONF_ALWAYS_SAVE_LATEST_FILE),
                s3_bucket=config.get(CONF_S3_BUCKET),
                camera_entity=camera.get(CONF_ENTITY_ID),
                name=camera.get(CONF_NAME),
                collection_id=collection_id,
            )
        )
    add_devices(entities)


class ObjectDetection(ImageProcessingEntity):
    """Perform object and label recognition."""

    def __init__(
        self,
        rekognition_client,
        s3_client,
        region,
        targets,
        confidence,
        roi_y_min,
        roi_x_min,
        roi_y_max,
        roi_x_max,
        scale,
        show_boxes,
        save_file_format,
        save_file_folder,
        save_timestamped_file,
        always_save_latest_file,
        s3_bucket,
        camera_entity,
        name=None,
        collection_id=None,
        
    ):
        """Init with the client."""
        self._collection_id = collection_id
        self._aws_rekognition_client = rekognition_client
        self._aws_s3_client = s3_client
        self._aws_region = region
        self._confidence = confidence
        self._targets = targets
        for target in self._targets:
            if CONF_CONFIDENCE not in target.keys():
                target.update({CONF_CONFIDENCE: self._confidence})
        self._targets_names = [target[CONF_TARGET] for target in targets]
        self._summary = {target: 0 for target in self._targets_names}

        self._camera_entity = camera_entity
        if name:  # Since name is optional.
            self._name = name
        else:
            entity_name = split_entity_id(camera_entity)[1]
            self._name = f"rekognition_{entity_name}"

        self._state = None  # The number of instances of interest
        self._last_detection = None  # The last time we detected a target
        self._objects = []  # The parsed raw data
        self._labels = []  # The parsed raw data
        self._targets_found = []  # The filtered targets data

        self._roi_dict = {
            "y_min": roi_y_min,
            "x_min": roi_x_min,
            "y_max": roi_y_max,
            "x_max": roi_x_max,
        }
        self._scale = scale
        self._show_boxes = show_boxes
        self._last_detection = None
        self._image_width = None
        self._image_height = None
        self._save_file_format = save_file_format
        self._save_file_folder = save_file_folder
        self._save_timestamped_file = save_timestamped_file
        self._always_save_latest_file = always_save_latest_file
        self._s3_bucket = s3_bucket
        self._image = None

    def process_image(self, image):
        """Process an image."""
        self._faces = [] 
        self._image = Image.open(io.BytesIO(bytearray(image)))  # used for saving only
        self._image_width, self._image_height = self._image.size

        # Resize image se necessario
        if self._scale != DEFAULT_SCALE:
            newsize = (self._image_width * self._scale, self._image_width * self._scale)
            self._image.thumbnail(newsize, Image.ANTIALIAS)
            self._image_width, self._image_height = self._image.size
            with io.BytesIO() as output:
                self._image.save(output, format="JPEG")
                image = output.getvalue()
            _LOGGER.debug(
                f"Image scaled with : {self._scale} W={self._image_width} H={self._image_height}"
            )

        self._state = None
        self._objects = []
        self._labels = []
        self._targets_found = []
        self._summary = {target: 0 for target in self._targets_names}
        saved_image_path = None

        # 🔹 1. Rilevamento oggetti per ottenere "Person"
        response_labels = self._aws_rekognition_client.detect_labels(Image={"Bytes": image})
        _LOGGER.debug("Rekognition detect_labels response: %s", response_labels)

        self._objects, self._labels = get_objects(response_labels)
        _LOGGER.debug("Parsed objects: %s", self._objects)
        _LOGGER.debug("Parsed labels: %s", self._labels)

        # 🔹 Resettiamo sempre le liste
        self._targets_found = []

        # 🔹 Manteniamo tutti gli oggetti nel sensore
        for obj in self._objects:
            if obj["confidence"] >= MIN_CONFIDENCE:
                self._targets_found.append(obj)

        # 🔹 Stato aggiornato con il numero totale di oggetti trovati
        self._state = len(self._targets_found)

        # 🔹 Controlla se almeno una persona è stata rilevata
        found_person = any(obj["name"] == "person" and obj["confidence"] >= MIN_CONFIDENCE for obj in self._objects)

        # 🔹 2. Rilevamento volti e confronto con la collection di Rekognition
        if found_person and self._collection_id:
            try:
                response_faces = self._aws_rekognition_client.search_faces_by_image(
                    CollectionId=self._collection_id,
                    Image={"Bytes": image},
                    MaxFaces=5,
                    FaceMatchThreshold=80
                )
                _LOGGER.debug("Rekognition search_faces_by_image response: %s", response_faces)

                self._faces = []  # 🔹 Assicura che l'elenco venga sempre aggiornato

                for match in response_faces.get("FaceMatches", []):
                    face = {
                        "name": match["Face"]["ExternalImageId"],
                        "confidence": round(match["Similarity"], 2),
                    }
                    self._faces.append(face)

                # 🔹 Se nessun volto viene rilevato, assicura che venga aggiornato a lista vuota
                if not self._faces:
                    _LOGGER.debug("Nessun volto riconosciuto, aggiornamento sensore con lista vuota.")
            except botocore.exceptions.ClientError as error:
                _LOGGER.error(f"Error in search_faces_by_image: {error}")

        # 🔹 Aggiorna il summary
        targets_found_names = [obj["name"] for obj in self._targets_found]
        self._summary = dict(Counter(targets_found_names))

        # 🔹 Aggiungi i volti riconosciuti nel summary
        for face in self._faces:
            self._summary[face["name"]] = self._summary.get(face["name"], 0) + 1

        if self._save_file_folder and (self._state > 0 or self._always_save_latest_file):
            saved_image_path = self.save_image(self._targets_found, self._save_file_folder)

        # 🔹 Fire events per oggetti rilevati
        for target in self._targets_found:
            target_event_data = target.copy()
            target_event_data[ATTR_ENTITY_ID] = self.entity_id
            if saved_image_path:
                target_event_data[SAVED_FILE] = saved_image_path
            self.hass.bus.fire(EVENT_OBJECT_DETECTED, target_event_data)

        # 🔹 Fire events per volti riconosciuti
        for face in self._faces:
            face_event_data = face.copy()
            face_event_data[ATTR_ENTITY_ID] = self.entity_id
            if saved_image_path:
                face_event_data[SAVED_FILE] = saved_image_path
            self.hass.bus.fire("rekognition.face_detected", face_event_data)




    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera_entity

    @property
    def state(self):
        """Return the state of the entity."""
        return self._state

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name

    @property
    def unit_of_measurement(self):
        """Return the unit of measurement."""
        return "targets"

    @property
    def should_poll(self):
        """Return the polling state."""
        return False

    @property
    def extra_state_attributes(self):
        """Return device specific state attributes."""
        attr = {}
        
        # Volti riconosciuti
        if hasattr(self, "_faces"):
            attr["recognized_faces"] = self._faces  # Lista dei volti riconosciuti con nome e confidenza
        
        # Targets rilevati
        attr["targets"] = self._targets
        attr["targets_found"] = [
            {obj["name"]: obj["confidence"]} for obj in self._targets_found
        ]
        
        # Riassunto rilevamenti
        attr["summary"] = self._summary
        
        if self._last_detection:
            attr["last_target_detection"] = self._last_detection
        
        attr["all_objects"] = [
            {obj["name"]: obj["confidence"]} for obj in self._objects
        ]
        
        # Parametri di salvataggio
        if self._save_file_folder:
            attr[CONF_SAVE_FILE_FORMAT] = self._save_file_format
            attr[CONF_SAVE_FILE_FOLDER] = str(self._save_file_folder)
            attr[CONF_SAVE_TIMESTAMPED_FILE] = self._save_timestamped_file
            attr[CONF_ALWAYS_SAVE_LATEST_FILE] = self._always_save_latest_file
            attr[CONF_SHOW_BOXES] = self._show_boxes
        
        # Se è configurato il bucket S3
        if self._s3_bucket:
            attr[CONF_S3_BUCKET] = self._s3_bucket
        
        # Etichette rilevate
        attr["labels"] = self._labels
        
        return attr


    def save_image(self, targets, directory) -> str:
        """Draw bounding boxes for detected objects and recognized faces, then save the image."""
        try:
            img = self._image.convert("RGB")
        except UnidentifiedImageError:
            _LOGGER.warning("Rekognition unable to process image, bad data")
            return None

        draw = ImageDraw.Draw(img)

        # Carica un font leggibile
        try:
            font = ImageFont.truetype("/usr/share/fonts/trfype/dejavu/DejaVuSans-Bold.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        # 🔹 Crea una mappa per abbinare i volti riconosciuti alle persone
        face_map = {}
        for face in self._faces:
            face_map[face["name"]] = face["confidence"]

        # 🔹 Disegna bounding box per le persone
        for obj in targets:
            if obj["name"] == "person":
                box = obj["bounding_box"]
                confidence = obj["confidence"]
                name_label = f"person: {confidence:.1f}%"

                # 🔹 Se il volto è stato riconosciuto, aggiunge il nome
                for face_name, face_conf in face_map.items():
                    name_label = f"{face_name}: {face_conf:.1f}%"

                # 🔹 Disegna bounding box della persona in rosso
                draw.rectangle(
                    [
                        (box["x_min"] * img.width, box["y_min"] * img.height),
                        (box["x_max"] * img.width, box["y_max"] * img.height),
                    ],
                    outline=RED,
                    width=3,
                )

                # 🔹 Aggiunge il testo sopra la persona
                draw.text(
                    (box["x_min"] * img.width, box["y_min"] * img.height - 10),
                    name_label,
                    fill="white",
                    font=font,
                )

        # 🔹 Disegna bounding box dei volti riconosciuti in giallo
        for face in self._faces:
            if "bounding_box" in face:
                box = face["bounding_box"]
                name = face["name"]
                confidence = face["confidence"]
                label = f"{name}: {confidence:.1f}%"

                draw.rectangle(
                    [
                        (box["x_min"] * img.width, box["y_min"] * img.height),
                        (box["x_max"] * img.width, box["y_max"] * img.height),
                    ],
                    outline=YELLOW,
                    width=2,
                )

                draw.text(
                    (box["x_min"] * img.width, box["y_min"] * img.height - 10),
                    label,
                    fill="yellow",
                    font=font,
                )

        # 🔹 Salvataggio immagine
        latest_save_path = (
            directory / f"{get_valid_filename(self._name).lower()}_latest.{self._save_file_format}"
        )
        img.save(latest_save_path)
        _LOGGER.info("Rekognition saved file %s", latest_save_path)
        
        return str(latest_save_path)



