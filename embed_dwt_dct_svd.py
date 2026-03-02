import numpy as np
import cv2
import pywt


class EmbedDwtDctSvd:

    def __init__(self, watermarks, scale=30, block=4):
        self._watermarks = watermarks
        self._wmLen = len(watermarks)
        self._scale = scale
        self._block = block

    # -----------------------------------
    # ENCODE
    # -----------------------------------

    def encode(self, image):

        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        Y = yuv[:, :, 0]

        # 🔹 2-Level DWT
        coeffs1 = pywt.dwt2(Y, 'haar')
        LL1, (LH1, HL1, HH1) = coeffs1

        coeffs2 = pywt.dwt2(LL1, 'haar')
        LL2, (LH2, HL2, HH2) = coeffs2

        # Embed in LH2 band (more robust)
        # Capacity check
        row, col = LH2.shape
        capacity = (row // self._block) * (col // self._block)

        if self._wmLen > capacity:
            raise Exception("Watermark too large for this image capacity")

        # Embed in LH2 band
        self._embed_frame(LH2)

        # Reconstruct
        LL1_recon = pywt.idwt2((LL2, (LH2, HL2, HH2)), 'haar')
        Y_recon = pywt.idwt2((LL1_recon, (LH1, HL1, HH1)), 'haar')

        Y_recon = np.clip(Y_recon, 0, 255)
        yuv[:, :, 0] = Y_recon.astype(np.uint8)

        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # -----------------------------------
    # DECODE
    # -----------------------------------

    def decode(self, image):

        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        Y = yuv[:, :, 0]

        coeffs1 = pywt.dwt2(Y, 'haar')
        LL1, _ = coeffs1

        coeffs2 = pywt.dwt2(LL1, 'haar')
        LL2, (LH2, _, _) = coeffs2

        return self._decode_frame(LH2)

    # -----------------------------------
    # EMBED FRAME
    # -----------------------------------

    def _embed_frame(self, frame):

        row, col = frame.shape
        num = 0

        for i in range(row // self._block):
            for j in range(col // self._block):

                block = frame[
                    i*self._block:(i+1)*self._block,
                    j*self._block:(j+1)*self._block
                ]

                bit = self._watermarks[num % self._wmLen]
                modified = self._embed_block(block, bit)

                frame[
                    i*self._block:(i+1)*self._block,
                    j*self._block:(j+1)*self._block
                ] = modified

                num += 1

    # -----------------------------------
    # EMBED BLOCK (Differential SVD)
    # -----------------------------------

    def _embed_block(self, block, bit):

        dct_block = cv2.dct(block.astype(np.float32))
        U, S, V = np.linalg.svd(dct_block)

        if bit == 1:
            S[0] = S[1] + self._scale
        else:
            S[0] = S[1] - self._scale

        modified = U @ np.diag(S) @ V
        return cv2.idct(modified)

    # -----------------------------------
    # DECODE FRAME
    # -----------------------------------

    def _decode_frame(self, frame):

        row, col = frame.shape
        scores = [[] for _ in range(self._wmLen)]
        num = 0

        for i in range(row // self._block):
            for j in range(col // self._block):

                block = frame[
                    i*self._block:(i+1)*self._block,
                    j*self._block:(j+1)*self._block
                ]

                bit = self._extract_block(block)
                wmIndex = num % self._wmLen
                scores[wmIndex].append(bit)

                num += 1

        avg = [np.mean(s) for s in scores]
        return (np.array(avg) > 0.5).astype(int).tolist()

    # -----------------------------------
    # EXTRACT BLOCK
    # -----------------------------------

    def _extract_block(self, block):

        dct_block = cv2.dct(block.astype(np.float32))
        _, S, _ = np.linalg.svd(dct_block)

        return 1 if S[0] > S[1] else 0