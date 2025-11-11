#!/usr/bin/env python3

from PIL import Image
import numpy as np
import os
import argparse


class LSBSteganography:
   
    def __init__(self):
        self.delimiter = "<<<END_OF_MESSAGE>>>"
        
    def _text_to_binary(self, text):
        binary = ''.join(format(ord(char), '08b') for char in text)
        return binary
    
    def _binary_to_text(self, binary):
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8: 
                text += chr(int(byte, 2))
        return text
    
    def _max_bytes_capacity(self, image):
        width, height = image.size
        # 3 kanala (RGB), svaki piksel može da sadrži 3 bita podataka
        total_bits = width * height * 3
        # Oduzimamo prostor za delimiter
        delimiter_bits = len(self.delimiter) * 8
        available_bits = total_bits - delimiter_bits
        return available_bits // 8
    
    def encode(self, image_path, message, output_path):
        try:
            img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            print("\n" + "="*60)
            print(f"KODIRANJE")
            print("\n" + "="*60)
            print( f"Poruka: {(message)} .")

            max_bytes = self._max_bytes_capacity(img)
            message_with_delimiter = message + self.delimiter
            
            if len(message_with_delimiter) > max_bytes:
                raise ValueError(
                    f"Poruka je prevelika! Maksimalno {max_bytes} bajtova, "
                    f"a poruka ima {len(message_with_delimiter)} bajtova."
                )
            
            binary_message = self._text_to_binary(message_with_delimiter)
            
            img_array = np.array(img)
            height, width, channels = img_array.shape
            
            flat_img = img_array.flatten()
            
            for i, bit in enumerate(binary_message):
                flat_img[i] = (flat_img[i] & 0xFE) | int(bit)
            
            encoded_img_array = flat_img.reshape((height, width, channels))
            
            encoded_img = Image.fromarray(encoded_img_array.astype('uint8'))
            encoded_img.save(output_path, 'PNG') 
            
            print(f"Poruka uspešno sakrivena u {output_path}")
            print(f"Dužina poruke: {len(message)} karaktera")
            print(f"Iskorišćenost kapaciteta: {len(message_with_delimiter)/max_bytes*100:.2f}%")
            
            return True
            
        except FileNotFoundError:
            print(f"Greška: Fajl '{image_path}' nije pronađen!")
            return False
        except Exception as e:
            print(f"Greška pri enkodovanju: {str(e)}")
            return False
    
    def decode(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            flat_img = img_array.flatten()
            
            # DEKODOVANJE: ekstraktujemo LSB iz svakog bajta
            binary_message = ''
            for byte in flat_img:
                binary_message += str(byte & 1)
            
            decoded_text = self._binary_to_text(binary_message)
            print("\n" + "="*60)
            print(f"DEKODIRANJE")
            print("\n" + "="*60)

            # Tražimo delimiter
            if self.delimiter in decoded_text:
                message = decoded_text.split(self.delimiter)[0]
                print(f"Poruka uspešno dekodovana!")
                print(f"Dužina poruke: {len(message)} karaktera")
                return message
            else:
                print("Nije pronađena sakrivena poruka u slici.")
                return None
                
        except FileNotFoundError:
            print(f"Greška: Fajl '{image_path}' nije pronađen!")
            return None
        except Exception as e:
            print(f"Greška pri dekodovanju: {str(e)}")
            return None
    
    def compare_images(self, original_path, encoded_path):
        """
        Poredi originalnu i enkodovanu sliku da pokaže razlike.
        """
        try:
            img1 = np.array(Image.open(original_path).convert('RGB'))
            img2 = np.array(Image.open(encoded_path).convert('RGB'))
            
            # Računanje razlika
            diff = np.abs(img1.astype(int) - img2.astype(int))
            total_pixels = img1.shape[0] * img1.shape[1] * img1.shape[2]
            changed_pixels = np.count_nonzero(diff)
            
            # MSE (Mean Squared Error) - mera kvaliteta
            mse = np.mean(diff ** 2)
            
            # PSNR (Peak Signal-to-Noise Ratio)
            if mse > 0:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            else:
                psnr = float('inf')
            
            print("\n" + "="*60)
            print("ANALIZA RAZLIKA IZMEĐU SLIKA")
            print("="*60)
            print(f"Ukupno piksela (sa RGB kanalima): {total_pixels:,}")
            print(f"Promenjenih piksela: {changed_pixels:,}")
            print(f"Procenat promene: {changed_pixels/total_pixels*100:.4f}%")
            print(f"MSE (Mean Squared Error): {mse:.6f}")
            print(f"PSNR (Peak SNR): {psnr:.2f} dB")
            
            if psnr > 40:
                print("Promene su vizuelno NEUOČLJIVE (PSNR > 40 dB)")
            else:
                print("Promene mogu biti VIDLJIVE (PSNR < 40 dB)")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"Greška pri poređenju: {str(e)}")


def create_demo_image():
    """
    Kreira demo sliku za testiranje.
    """
    width, height = 800, 600
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [
                int(255 * x / width),      # R
                int(255 * y / height),      # G
                int(128 + 127 * np.sin(x/50))  # B 
            ]
    
    img = Image.fromarray(img_array)
    img.save('demo_image.png')
    print("Kreirana demo slika: demo_image.png")


def main():
    parser = argparse.ArgumentParser(
        description='LSB Steganografija - Skrivanje poruka u slike',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Primeri upotrebe:

  Enkodovanje poruke
  python steganography.py encode input.png "Tajna poruka!" output.png

  Dekodovanje poruke
  python steganography.py decode output.png

  Poređenje slika
  python steganography.py compare input.png output.png

  Kreiranje demo slike
  python steganography.py demo
        """
    )
    
    parser.add_argument('action', choices=['encode', 'decode', 'compare', 'demo'],
                       help='Akcija koju želiš da izvrsiš')
    parser.add_argument('files', nargs='*', help='Putanje do fajlova')
    
    args = parser.parse_args()
    stego = LSBSteganography()
    
    if args.action == 'demo':
        print("\n" + "="*60)
        print("DEMO MOD - Kreiranje test slike i enkodovanje")
        print("="*60 + "\n")
        
        # Kreiranje demo slike
        create_demo_image()
        
        # Test poruke
        test_messages = [
            "Zdravo, ovo je tajna poruka!",
            "Implementacija LSB algoritma za skrivanje poruka. Testni tekst je Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centu",
        ]
        
        for i, msg in enumerate(test_messages, 1):
            output = f'encoded_demo_{i}.png'
            print(f"\n--- Test {i} ---")
            stego.encode('demo_image.png', msg, output)
            decoded = stego.decode(output)
            print(f"Dekodovana poruka: '{decoded}'")
            
            if decoded == msg:
                print("USPEŠNO: Poruka se potpuno poklapa!")
            else:
                print("REŠKA: Poruke se ne poklapaju!")
        
        # Poređenje
        print("\n")
        stego.compare_images('demo_image.png', 'encoded_demo_2.png')
    
    elif args.action == 'encode':
        if len(args.files) < 3:
            print("Potrebna su 3 argumenta: input_image message output_image")
            return
        stego.encode(args.files[0], args.files[1], args.files[2])
    
    elif args.action == 'decode':
        if len(args.files) < 1:
            print("Potreban je argument: image_path")
            return
        message = stego.decode(args.files[0])
        if message:
            print(f"\nDekodovana poruka:\n{message}")
    
    elif args.action == 'compare':
        if len(args.files) < 2:
            print("Potrebna su 2 argumenta: original_image encoded_image")
            return
        stego.compare_images(args.files[0], args.files[1])


if __name__ == "__main__":
    main()