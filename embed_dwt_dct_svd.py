import numpy as np
import cv2
import pywt


class EmbedDwtDctSvd:

    def __init__(self, watermarks, scales=None, block=4):
        self._watermarks = watermarks
        self._wmLen = len(watermarks)
        self._scales = scales if scales else [0, 36, 0]
        self._block = block

    def encode(self, image):
        row, col, _ = image.shape
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            sub = yuv[:row//4*4, :col//4*4, channel]
            ca1, (h1, v1, d1) = pywt.dwt2(sub, 'haar')

            self._encode_frame(ca1, self._scales[channel])

            recon = pywt.idwt2((ca1, (h1, v1, d1)), 'haar')
            recon = np.clip(recon, 0, 255)

            yuv[:row//4*4, :col//4*4, channel] = recon.astype(np.uint8)

        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def decode(self, image):
        row, col, _ = image.shape
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        scores = [[] for _ in range(self._wmLen)]

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, _ = pywt.dwt2(yuv[:row//4*4, :col//4*4, channel], 'haar')
            scores = self._decode_frame(ca1, self._scales[channel], scores)

        avgScores = list(map(lambda l: np.array(l).mean(), scores))
        bits = (np.array(avgScores) > 0.5)

        return bits.astype(int).tolist()

    # ---------- Internal ----------

    def _encode_frame(self, frame, scale):
        row, col = frame.shape
        num = 0

        for i in range(row//self._block):
            for j in range(col//self._block):
                block = frame[i*self._block:(i+1)*self._block,
                              j*self._block:(j+1)*self._block]

                wmBit = self._watermarks[num % self._wmLen]
                modified = self._embed_block(block, wmBit, scale)

                frame[i*self._block:(i+1)*self._block,
                      j*self._block:(j+1)*self._block] = modified

                num += 1

    def _embed_block(self, block, wmBit, scale):
        dct_block = cv2.dct(block.astype(np.float32))
        u, s, v = np.linalg.svd(dct_block)

        q = np.floor(s[0] / scale)
        s[0] = (q + 0.25 + 0.5 * wmBit) * scale

        modified = np.dot(u, np.dot(np.diag(s), v))
        return cv2.idct(modified)

    def _decode_frame(self, frame, scale, scores):
        row, col = frame.shape
        num = 0

        for i in range(row//self._block):
            for j in range(col//self._block):
                block = frame[i*self._block:(i+1)*self._block,
                              j*self._block:(j+1)*self._block]

                bit = self._extract_block(block, scale)
                wmBit = num % self._wmLen
                scores[wmBit].append(bit)

                num += 1

        return scores

    def _extract_block(self, block, scale):
        dct_block = cv2.dct(block.astype(np.float32))
        _, s, _ = np.linalg.svd(dct_block)
        return int((s[0] % scale) > scale * 0.5)