import struct

def clean_hexdump(hex_text):
    cleaned_hex = ""
    lines = hex_text.strip().split('\n')
    for line in lines:
        # Remove the offset (e.g. "00000000: ") and trailing whitespace
        if ':' in line:
            parts = line.split(':')
            if len(parts) > 1:
                hex_content = parts[1].strip()
                # Remove the spaces in the middle
                cleaned_hex += hex_content.replace(" ", "")
    
    # Convert the hex string to bytes
    return bytes.fromhex(cleaned_hex)

def parse_jpeg_structure(data):
    MARKERS = {
        0xD8: "SOI (Start of Image)",
        0xE0: "APP0 (JFIF Header)",
        0xDB: "DQT (Define Quantization Table)",
        0xC0: "SOF0 (Start of Frame, Baseline DCT)",
        0xC4: "DHT (Define Huffman Table)",
        0xDA: "SOS (Start of Scan)",
        0xD9: "EOI (End of Image)"
    }
    
    print(f"{'OFFSET (Hex)':<15} | {'MARKER':<8} | {'SEGMENT NAME':<35} | {'LENGTH':<8} | {'DETAILS'}")
    print("-" * 100)
    
    i = 0
    size = len(data)
    
    while i < size:
        if data[i] == 0xFF:
            # Handle Padding (FF FF)
            if i + 1 < size and data[i+1] == 0xFF:
                i += 1
                continue
            
            marker = data[i+1]
            if marker == 0x00: # Byte stuffing inside ECS
                i += 2
                continue
                
            offset_str = f"0x{i:04X}"
            marker_str = f"FF {marker:02X}"
            name_str = MARKERS.get(marker, f"Unknown (FF {marker:02X})")
            
            # 1. Marker without length parameter
            if marker in [0xD8, 0xD9, 0x01]: 
                print(f"{offset_str:<15} | {marker_str:<8} | {name_str:<35} | {'-':<8} | Marker only")
                i += 2
            
            # 2. Marker with length parameter
            else:
                length = struct.unpack(">H", data[i+2:i+4])[0]
                payload = data[i+4 : i+2+length]
                details = ""
                
                if marker == 0xE0: # APP0
                    identifier = payload[:4].decode('ascii', errors='ignore')
                    details = f"ID: {identifier}"
                elif marker == 0xDB: # DQT
                    # DQT may contain multiple tables, each 65 bytes (1 byte info + 64 bytes data)
                    qty = (length - 2) // 65
                    ids = []
                    for q in range(qty):
                        table_id = payload[q*65] & 0x0F
                        ids.append(str(table_id))
                    details = f"Table ID(s): {', '.join(ids)}"
                elif marker == 0xC0: # SOF0
                    h, w = struct.unpack(">HH", payload[1:5])
                    details = f"Size: {w}x{h}"
                elif marker == 0xC4: # DHT
                    # Huffman Table info: bit 4=class(0:DC,1:AC), bit 0-3=ID
                    ht_info = payload[0]
                    ht_class = "AC" if (ht_info >> 4) else "DC"
                    ht_id = ht_info & 0x0F
                    details = f"Type: {ht_class}, ID: {ht_id}"
                
                print(f"{offset_str:<15} | {marker_str:<8} | {name_str:<35} | {length:<8} | {details}")
                
                # 3. SOS special handling: the following is compressed data (ECS)
                if marker == 0xDA:
                    scan_start = i + 2 + length
                    scan_end = scan_start
                    # Find the next Marker (EOI)
                    while scan_end < size - 1:
                        if data[scan_end] == 0xFF and data[scan_end+1] != 0x00:
                            break
                        scan_end += 1
                    
                    ecs_len = scan_end - scan_start
                    print(f"{f'0x{scan_start:04X}':<15} | {'-':<8} | {'ECS (Entropy Coded Segment)':<35} | {ecs_len:<8} | Compressed Data")
                    i = scan_end
                    continue

                i += 2 + length
        else:
            i += 1

hex_input = """
00000000: FFD8 FFE0 0010 4A46 4946 0001 0100 0001 
00000010: 0001 0000 FFDB 0043 0003 0202 0302 0203 
00000020: 0303 0304 0303 0405 0805 0504 0405 0A07 
00000030: 0706 080C 0A0C 0C0B 0A0B 0B0D 0E12 100D 
00000040: 0E11 0E0B 0B10 1610 1113 1415 1515 0C0F 
00000050: 1718 1614 1812 1415 14FF DB00 4301 0304 
00000060: 0405 0405 0905 0509 140D 0B0D 1414 1414 
00000070: 1414 1414 1414 1414 1414 1414 1414 1414 
00000080: 1414 1414 1414 1414 1414 1414 1414 1414 
00000090: 1414 1414 1414 1414 1414 1414 1414 FFC0 
000000A0: 0011 0800 1000 1003 0122 0002 1101 0311 
000000B0: 01FF C400 1F00 0001 0501 0101 0101 0100 
000000C0: 0000 0000 0000 0001 0203 0405 0607 0809 
000000D0: 0A0B FFC4 00B5 1000 0201 0303 0204 0305 
000000E0: 0504 0400 0001 7D01 0203 0004 1105 1221 
000000F0: 3141 0613 5161 0722 7114 3281 91A1 0823 
00000100: 42B1 C115 52D1 F024 3362 7282 090A 1617 
00000110: 1819 1A25 2627 2829 2A34 3536 3738 393A 
00000120: 4344 4546 4748 494A 5354 5556 5758 595A 
00000130: 6364 6566 6768 696A 7374 7576 7778 797A 
00000140: 8384 8586 8788 898A 9293 9495 9697 9899 
00000150: 9AA2 A3A4 A5A6 A7A8 A9AA B2B3 B4B5 B6B7 
00000160: B8B9 BAC2 C3C4 C5C6 C7C8 C9CA D2D3 D4D5 
00000170: D6D7 D8D9 DAE1 E2E3 E4E5 E6E7 E8E9 EAF1 
00000180: F2F3 F4F5 F6F7 F8F9 FAFF C400 1F01 0003 
00000190: 0101 0101 0101 0101 0100 0000 0000 0001 
000001A0: 0203 0405 0607 0809 0A0B FFC4 00B5 1100 
000001B0: 0201 0204 0403 0407 0504 0400 0102 7700 
000001C0: 0102 0311 0405 2131 0612 4151 0761 7113 
000001D0: 2232 8108 1442 91A1 B1C1 0923 3352 F015 
000001E0: 6272 D10A 1624 34E1 25F1 1718 191A 2627 
000001F0: 2829 2A35 3637 3839 3A43 4445 4647 4849 
00000200: 4A53 5455 5657 5859 5A63 6465 6667 6869 
00000210: 6A73 7475 7677 7879 7A82 8384 8586 8788 
00000220: 898A 9293 9495 9697 9899 9AA2 A3A4 A5A6 
00000230: A7A8 A9AA B2B3 B4B5 B6B7 B8B9 BAC2 C3C4 
00000240: C5C6 C7C8 C9CA D2D3 D4D5 D6D7 D8D9 DAE2 
00000250: E3E4 E5E6 E7E8 E9EA F2F3 F4F5 F6F7 F8F9 
00000260: FAFF DA00 0C03 0100 0211 0311 003F 00F0 
00000270: 6D6B E36E A9E3 3D62 E7FB 26DE 0D33 4167 
00000280: 6B89 2497 721F 2D4F CC47 CA59 76B3 7CCC 
00000290: 7F77 CFCD B704 8E13 C5BE 2587 C49A 8C5A 
000002A0: 6E99 7EFA BEA7 14CC 2692 0DB2 C4C4 124B 
000002B0: 2346 CCAC 71C7 1D94 6335 EB1F 01E5 F0BE 
000002C0: 8770 D1EA 7A78 BA3B 99E3 9121 1333 48CD 
000002D0: 8CFC E768 2034 801C 7019 BD5B 3EB0 7E11 
000002E0: E95E 1BBA 7D4A E6D2 2D1A C203 F688 6CE3 
000002F0: 8630 586E 3CED 4016 3396 EC01 EDD4 9200 
00000300: 3FFF D9 
"""

# Execute
binary_data = clean_hexdump(hex_input)
parse_jpeg_structure(binary_data)