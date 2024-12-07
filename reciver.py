import asyncio
import cv2
import numpy as np
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay

class VideoReceiverTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()  # initialize parent class
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Here, you can process the frame as needed
        # For now, we just display it
        cv2.imshow("Received Frame", img)
        cv2.waitKey(1)
        
        return frame

async def offer(pc, offer_sdp):
    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc.localDescription

async def main():
    # Setup RTC connection
    pc = RTCPeerConnection()
    print("Created RTC peer connection")
    
    # Assuming you already have an offer SDP (you would typically get it from a signaling server)
    offer_sdp = RTCSessionDescription(sdp="YOUR_OFFER_SDP", type="offer")
    
    # Set up the connection and receive the video
    pc.on('iceconnectionstatechange', lambda: print('ICE connection state is {}'.format(pc.iceConnectionState)))

    @pc.on('track')
    def on_track(track):
        if track.kind == 'video':
            print("Receiving video track")
            video_track = VideoReceiverTrack(track)
            pc.addTrack(video_track)
    
    # Create the answer and set local description
    local_desc = await offer(pc, offer_sdp)
    print("Generated answer SDP")

    # Keep the program running to handle incoming frames
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
