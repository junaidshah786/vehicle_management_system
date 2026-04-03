"""
Vehicle Plate Cache Service
Maintains an in-memory cache of all vehicle registration numbers,
synced in real-time via Firestore on_snapshot listener.
Provides hybrid fuzzy matching for OCR error correction.
"""

import logging
import threading
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# ============ OCR CONFUSION MAP ============
# Characters commonly confused by OCR engines
# Maps a character to the list of characters it's commonly misread as
OCR_CONFUSIONS: Dict[str, list] = {
    "0": ["O", "Q", "D"],
    "O": ["0", "Q", "D"],
    "1": ["I", "L", "T"],
    "I": ["1", "L", "T"],
    "5": ["S"],
    "S": ["5"],
    "8": ["B"],
    "B": ["8"],
    "2": ["Z"],
    "Z": ["2"],
    "6": ["G"],
    "G": ["6"],
    "C": ["0"],  # C often read as 0 (your exact case)
}


class VehiclePlateCache:
    """
    Singleton in-memory cache of vehicle registration plates.
    Auto-syncs with Firestore via real-time listener.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Main cache: {clean_registration_number: firestore_doc_id}
        self._plate_to_id: Dict[str, str] = {}
        # Reverse cache for quick lookups: {firestore_doc_id: clean_registration_number}
        self._id_to_plate: Dict[str, str] = {}
        # Firestore listener unsubscribe handle
        self._unsubscribe = None

        logger.info("VehiclePlateCache initialized (empty, awaiting Firestore sync)")

    def start_listener(self, db):
        """Start Firestore real-time listener on vehicleDetails collection."""
        if self._unsubscribe is not None:
            logger.warning("Firestore listener already active, skipping duplicate start")
            return

        logger.info("Starting Firestore real-time listener for vehicle plates cache...")

        def on_snapshot(col_snapshot, changes, read_time):
            for change in changes:
                doc = change.document
                doc_id = doc.id
                data = doc.to_dict()
                reg = (data.get("registrationNumber") or "").replace(" ", "").upper()

                if change.type.name == "REMOVED":
                    # Vehicle deleted
                    old_plate = self._id_to_plate.pop(doc_id, None)
                    if old_plate:
                        self._plate_to_id.pop(old_plate, None)
                    logger.debug(f"Cache REMOVE: {old_plate} (doc {doc_id})")

                else:
                    # ADDED or MODIFIED
                    # If modified, remove old plate mapping first
                    old_plate = self._id_to_plate.get(doc_id)
                    if old_plate and old_plate != reg:
                        self._plate_to_id.pop(old_plate, None)

                    if reg:
                        self._plate_to_id[reg] = doc_id
                        self._id_to_plate[doc_id] = reg

                    action = "ADD" if change.type.name == "ADDED" else "UPDATE"
                    logger.debug(f"Cache {action}: {reg} -> {doc_id}")

            logger.info(f"Plate cache synced: {len(self._plate_to_id)} vehicles loaded")

        self._unsubscribe = db.collection("vehicleDetails").on_snapshot(on_snapshot)
        logger.info("Firestore real-time listener started for vehicle plates cache")

    def stop_listener(self):
        """Stop Firestore listener (for graceful shutdown)."""
        if self._unsubscribe:
            self._unsubscribe.unsubscribe()
            self._unsubscribe = None
            logger.info("Firestore listener stopped")

    @property
    def size(self) -> int:
        """Number of vehicles in cache."""
        return len(self._plate_to_id)

    def exact_lookup(self, plate: str) -> Optional[str]:
        """Exact match lookup. Returns Firestore doc_id or None."""
        clean = plate.replace(" ", "").upper()
        return self._plate_to_id.get(clean)

    def get_all_plates(self) -> set:
        """Get all cached plates as a set (for fuzzy matching)."""
        return set(self._plate_to_id.keys())

    # ============ HYBRID FUZZY MATCHING ============

    def fuzzy_match(self, raw_text: str) -> Optional[Tuple[str, str, str]]:
        """
        Hybrid fuzzy match against cached plates.
        Returns (matched_plate, doc_id, match_method) or None.

        Strategy:
          1. OCR-aware substitutions (fast, targeted)
          2. Levenshtein distance=1 fallback (broad, catches anything)

        Safety: Only returns a match if EXACTLY 1 candidate is found.
        """
        if not raw_text:
            return None

        clean = raw_text.replace(" ", "").upper()

        # Quick check: exact match (shouldn't get here, but just in case)
        doc_id = self._plate_to_id.get(clean)
        if doc_id:
            return (clean, doc_id, "exact")

        # Step 1: OCR-aware substitutions
        result = self._ocr_aware_match(clean)
        if result:
            return result

        # Step 2: Levenshtein distance=1 fallback
        result = self._levenshtein_match(clean)
        if result:
            return result

        return None

    def _ocr_aware_match(self, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Generate all OCR-confusion variants and check against cache.
        Only substitutes characters that have known OCR confusions.
        """
        # Find which positions have possible OCR confusions
        ambiguous_positions = []
        for i, char in enumerate(text):
            if char in OCR_CONFUSIONS:
                ambiguous_positions.append(i)

        if not ambiguous_positions:
            return None

        # Limit to avoid combinatorial explosion (max 6 ambiguous chars = 64 combos max)
        if len(ambiguous_positions) > 6:
            ambiguous_positions = ambiguous_positions[:6]

        # Generate all variants using iterative approach
        variants = {text}
        for pos in ambiguous_positions:
            char = text[pos]
            new_variants = set()
            for variant in variants:
                for replacement in OCR_CONFUSIONS[char]:
                    new_variant = variant[:pos] + replacement + variant[pos + 1:]
                    new_variants.add(new_variant)
            variants.update(new_variants)

        # Remove the original text (already checked as exact match)
        variants.discard(text)

        # Check each variant against cache
        matches = []
        for variant in variants:
            doc_id = self._plate_to_id.get(variant)
            if doc_id:
                matches.append((variant, doc_id))

        if len(matches) == 1:
            plate, doc_id = matches[0]
            logger.info(f"FUZZY MATCH (ocr_aware): '{text}' -> '{plate}' (doc: {doc_id})")
            return (plate, doc_id, "ocr_aware")
        elif len(matches) > 1:
            logger.warning(
                f"FUZZY MATCH AMBIGUOUS (ocr_aware): '{text}' matched {len(matches)} vehicles: "
                f"{[m[0] for m in matches]}. Rejecting."
            )
            return None

        return None

    def _levenshtein_match(self, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Fallback: check all cached plates for Levenshtein distance = 1.
        Only accepts if exactly 1 plate matches.
        """
        matches = []
        for plate, doc_id in self._plate_to_id.items():
            if self._levenshtein_distance_max1(text, plate):
                matches.append((plate, doc_id))

            # Early exit if we already found 2+ matches (ambiguous)
            if len(matches) > 1:
                break

        if len(matches) == 1:
            plate, doc_id = matches[0]
            logger.info(f"FUZZY MATCH (levenshtein): '{text}' -> '{plate}' (doc: {doc_id})")
            return (plate, doc_id, "levenshtein")
        elif len(matches) > 1:
            logger.warning(
                f"FUZZY MATCH AMBIGUOUS (levenshtein): '{text}' matched 2+ vehicles. Rejecting."
            )
            return None

        return None

    @staticmethod
    def _levenshtein_distance_max1(s1: str, s2: str) -> bool:
        """
        Optimized check: is Levenshtein distance between s1 and s2 exactly 1?
        Handles substitution, insertion, and deletion.
        Returns True if distance == 1, False otherwise.
        Much faster than computing full distance — O(n) time.
        """
        len1, len2 = len(s1), len(s2)
        diff = abs(len1 - len2)

        if diff > 1:
            return False

        if diff == 0:
            # Same length: check for exactly 1 substitution
            mismatches = sum(1 for a, b in zip(s1, s2) if a != b)
            return mismatches == 1

        # Length differs by 1: check for exactly 1 insertion/deletion
        longer, shorter = (s1, s2) if len1 > len2 else (s2, s1)
        i = j = 0
        found_diff = False
        while i < len(longer) and j < len(shorter):
            if longer[i] != shorter[j]:
                if found_diff:
                    return False
                found_diff = True
                i += 1  # Skip the extra character in longer string
            else:
                i += 1
                j += 1
        return True


# Singleton instance
plate_cache = VehiclePlateCache()
