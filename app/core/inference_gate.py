import asyncio

# Live synchronous ML inference (person detection -> silhouette extraction ->
# gait/appearance encoding -> gallery matching) used to be called directly
# inside async FastAPI route handlers, blocking the single event loop for the
# full duration of each call - health, auth, and every other request froze
# while any inference ran.
#
# Moving those calls onto a worker thread (asyncio.to_thread) fixes that, but
# introduces a real hazard that didn't exist before: GaitService.enroll_images
# mutates shared, unlocked instance state (self.gallery_features,
# self.gallery_labels, self.appearance_gallery_features/_labels) in place
# before saving it - safe only because direct synchronous calls happened to
# serialize on the event loop. Once genuinely offloaded to threads, two
# concurrent enrollments could race on that same mutable state.
#
# Until that state is made properly thread-safe and verified under real
# concurrent load, this semaphore keeps exactly one live inference call
# in flight at a time - matching today's de facto behavior while letting
# the event loop stay free for unrelated requests during inference. Raise
# MAX_CONCURRENT_LIVE_INFERENCE only after both (a) the shared-state race
# above is fixed and (b) real concurrent load testing proves it safe.
MAX_CONCURRENT_LIVE_INFERENCE = 1

_inference_gate = asyncio.Semaphore(MAX_CONCURRENT_LIVE_INFERENCE)


def get_inference_gate() -> asyncio.Semaphore:
    return _inference_gate
