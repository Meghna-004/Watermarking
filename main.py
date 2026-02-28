# from watermark_system import WatermarkSystem


# if __name__ == "__main__":

#     system = WatermarkSystem()

#     uid = system.generate_user_watermark("signature.png")
#     print("Generated Unique ID:", uid)

#     system.embed("./assets/img.png", "watermarked_output.jpg")
#     print("Watermark embedded successfully!")

#     extracted = system.decode("./assets/tampered.png")
#     print("Extracted bits:", len(extracted))

    

from watermark_system import WatermarkSystem

if __name__ == "__main__":

    system = WatermarkSystem()

    uid = system.generate_user_watermark(
        "./assets/img_1.png",
        "./assets/img.png"
    )

    print("Generated Unique ID:", uid)

    system.embed("./assets/img.png", "./assets/watermarked_output.jpg")
    print("Watermark embedded successfully!")

    # Check original
    result1 = system.verify("./assets/watermarked_output.jpg")
    print("Original Image:", result1)

    # Check tampered
    result2 = system.verify("./assets/tampered.png")
    print("Tampered Image:", result2)