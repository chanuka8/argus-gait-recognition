"""ONVIF client for enterprise camera integration."""

import re
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

from monitoring.logging_config import get_logger


class ONVIFProfile:
    """Parsed ONVIF media profile."""

    def __init__(self, name: str, token: str, stream_uri: str = "", resolution: tuple = (640, 480), encoding: str = "H264") -> None:
        self.name = name
        self.token = token
        self.stream_uri = stream_uri
        self.resolution = resolution
        self.encoding = encoding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "token": self.token,
            "stream_uri": self.stream_uri,
            "resolution": self.resolution,
            "encoding": self.encoding,
        }


class ONVIFCapabilities:
    """Parsed ONVIF device capabilities."""

    def __init__(self) -> None:
        self.has_ptz: bool = False
        self.has_media: bool = False
        self.has_imaging: bool = False
        self.has_analytics: bool = False
        self.device_info: Dict[str, str] = {}
        self.profiles: List[ONVIFProfile] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_ptz": self.has_ptz,
            "has_media": self.has_media,
            "has_imaging": self.has_imaging,
            "has_analytics": self.has_analytics,
            "device_info": self.device_info,
            "profiles": [p.to_dict() for p in self.profiles],
        }


class ONVIFClient:
    """ONVIF camera discovery and capability client.

    Performs ONVIF device interrogation via SOAP/XML parsing.
    Does not require zeep or onvif-py; uses lightweight XML parsing
    for environments where those dependencies are unavailable.
    """

    NAMESPACES = {
        "s": "http://www.w3.org/2003/05/soap-envelope",
        "tds": "http://www.onvif.org/ver10/device/wsdl",
        "trt": "http://www.onvif.org/ver10/media/wsdl",
        "tt": "http://www.onvif.org/ver10/schema",
        "d": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
        "wsadis": "http://schemas.xmlsoap.org/ws/2004/08/addressing",
    }

    def __init__(self, host: str, port: int = 80, username: str = "", password: str = "") -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self._logger = get_logger("onvif_client")
        self._device_url = f"http://{host}:{port}/onvif/device_service"
        self._media_url = f"http://{host}:{port}/onvif/media_service"

    def build_soap_envelope(self, body_xml: str) -> str:
        """Build SOAP envelope with optional WS-Security header."""
        header = ""
        if self.username:
            header = f"""<s:Header>
  <Security xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
    <UsernameToken>
      <Username>{self.username}</Username>
      <Password>{self.password}</Password>
    </UsernameToken>
  </Security>
</s:Header>"""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
{header}
<s:Body>
{body_xml}
</s:Body>
</s:Envelope>"""

    def parse_capabilities_response(self, xml_text: str) -> ONVIFCapabilities:
        """Parse a GetCapabilities SOAP response into ONVIFCapabilities."""
        caps = ONVIFCapabilities()
        try:
            root = ElementTree.fromstring(xml_text)
            body = root.find(".//{http://www.w3.org/2003/05/soap-envelope}Body")
            if body is None:
                return caps

            caps_elem = body.find(".//{http://www.onvif.org/ver10/device/wsdl}Capabilities")
            if caps_elem is None:
                caps_elem = body.find(".//{http://www.onvif.org/ver10/schema}Capabilities")
            if caps_elem is None:
                return caps

            if caps_elem.find(".//{http://www.onvif.org/ver10/schema}PTZ") is not None:
                caps.has_ptz = True
            if caps_elem.find(".//{http://www.onvif.org/ver10/schema}Media") is not None:
                caps.has_media = True
            if caps_elem.find(".//{http://www.onvif.org/ver10/schema}Imaging") is not None:
                caps.has_imaging = True
            if caps_elem.find(".//{http://www.onvif.org/ver10/schema}Analytics") is not None:
                caps.has_analytics = True

        except ElementTree.ParseError as e:
            self._logger.error(f"Failed to parse ONVIF capabilities: {e}")

        return caps

    def parse_device_info_response(self, xml_text: str) -> Dict[str, str]:
        """Parse GetDeviceInformation SOAP response."""
        info: Dict[str, str] = {}
        try:
            root = ElementTree.fromstring(xml_text)
            body = root.find(".//{http://www.w3.org/2003/05/soap-envelope}Body")
            if body is None:
                return info

            resp = body.find(".//{http://www.onvif.org/ver10/device/wsdl}GetDeviceInformationResponse")
            if resp is None:
                return info

            for field in ["Manufacturer", "Model", "FirmwareVersion", "SerialNumber", "HardwareId"]:
                elem = resp.find(f"{{http://www.onvif.org/ver10/device/wsdl}}{field}")
                if elem is not None and elem.text:
                    info[field.lower()] = elem.text

        except ElementTree.ParseError as e:
            self._logger.error(f"Failed to parse device info: {e}")

        return info

    def parse_profiles_response(self, xml_text: str) -> List[ONVIFProfile]:
        """Parse GetProfiles SOAP response into list of ONVIFProfile."""
        profiles: List[ONVIFProfile] = []
        try:
            root = ElementTree.fromstring(xml_text)
            body = root.find(".//{http://www.w3.org/2003/05/soap-envelope}Body")
            if body is None:
                return profiles

            for p_elem in body.findall(".//{http://www.onvif.org/ver10/media/wsdl}Profiles"):
                token = p_elem.attrib.get("token", "")
                name_elem = p_elem.find("{http://www.onvif.org/ver10/schema}Name")
                name = name_elem.text if name_elem is not None and name_elem.text else token

                width, height = 640, 480
                enc = "H264"

                vec = p_elem.find(".//{http://www.onvif.org/ver10/schema}Resolution")
                if vec is not None:
                    w_elem = vec.find("{http://www.onvif.org/ver10/schema}Width")
                    h_elem = vec.find("{http://www.onvif.org/ver10/schema}Height")
                    if w_elem is not None and w_elem.text:
                        width = int(w_elem.text)
                    if h_elem is not None and h_elem.text:
                        height = int(h_elem.text)

                enc_elem = p_elem.find(".//{http://www.onvif.org/ver10/schema}Encoding")
                if enc_elem is not None and enc_elem.text:
                    enc = enc_elem.text

                profiles.append(ONVIFProfile(
                    name=name,
                    token=token,
                    resolution=(width, height),
                    encoding=enc,
                ))

        except ElementTree.ParseError as e:
            self._logger.error(f"Failed to parse profiles: {e}")

        return profiles

    @staticmethod
    def build_rtsp_url(host: str, port: int = 554, path: str = "/Streaming/Channels/101", username: str = "", password: str = "") -> str:
        """Build authenticated RTSP URL."""
        auth = f"{username}:{password}@" if username else ""
        return f"rtsp://{auth}{host}:{port}{path}"

    @staticmethod
    def extract_host_from_xaddr(xaddr: str) -> Optional[str]:
        """Extract host IP from ONVIF XAddr URL."""
        match = re.search(r"https?://([^:/]+)", xaddr)
        return match.group(1) if match else None
