from scapy.all import sniff
import pandas as pd
import numpy as np

def extract_basic_features(packet):
    features = {}

    features["Protocol"] = packet.proto if hasattr(packet, "proto") else 0
    features["Packet Length"] = len(packet)
    features["TTL"] = packet.ttl if hasattr(packet, "ttl") else 0

    return features


def capture_packets(duration=10):
    packets = sniff(timeout=duration)

    feature_list = []

    for pkt in packets:
        try:
            feat = extract_basic_features(pkt)
            feature_list.append(feat)
        except:
            continue

    df = pd.DataFrame(feature_list)

    return df