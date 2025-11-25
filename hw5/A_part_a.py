import numpy as np
from decimal import Decimal, getcontext

def main():
    getcontext().prec = 100
    # -5:0.001;-4:0.002;-3:0.004;-2:0.020;-1:0.040;0:0.86;1:0.040;2:0.020;3:0.004;4:0.002;5:0.001;EOB:0.006
    # a:0.001;b:0.002;c:0.004;d:0.020;e:0.040;f:0.86;g:0.040;h:0.020;i:0.004;j:0.002;k:0.001;EOB:0.006
    # -5:a; -4:b; -3:c; -2:d; -1:e; 0:f; 1:g; 2:h; 3:i; 4:j; 5:k; EOB:EOB
    symbol_probs = {
        -5: Decimal('0.001'),
        -4: Decimal('0.002'),
        -3: Decimal('0.004'),
        -2: Decimal('0.020'),
        -1: Decimal('0.040'),
        0:  Decimal('0.86'),
        1:  Decimal('0.040'),
        2:  Decimal('0.020'),
        3:  Decimal('0.004'),
        4:  Decimal('0.002'),
        5:  Decimal('0.001'),
        'EOB': Decimal('0.006')
    }

    sorted_keys = sorted([k for k in symbol_probs.keys() if k != 'EOB']) + ['EOB']
    
    cdf = {}
    current_low = Decimal(0)
    for sym in sorted_keys:
        prob = symbol_probs[sym]
        cdf[sym] = (current_low, current_low + prob)
        current_low += prob

    if current_low != Decimal(1):
        raise ValueError(f"The probability sum is not 1: {current_low}")

    # The quantized image block (8x8)
    block = np.array([
        [-19, -3,  0,  1,  0,  0,  0,  0],
        [  3,  1,  2,  0,  0,  0,  0,  0],
        [  0,  0, -1,  0,  0,  0,  0,  0],
        [  2,  0,  1,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0]
    ])

    # Execute the Zig-Zag scan
    # Standard JPEG Zig-Zag index path
    zigzag_indices = [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
        (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
        (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
        (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
        (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
        (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
        (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
        (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]
    # -3 3 0 1 0 1 2 0 2 0 0 -1 0 0 0 0 0 1 EOB
    # c i f g f g h f h f f e f f f f f g EOB
    # Extract the Zig-Zag sequence
    zigzag_sequence = [block[r, c] for r, c in zigzag_indices]

    # Prepare the encoding sequence
    # (a) Remove the first DC coefficient (index 0)
    ac_sequence = zigzag_sequence[1:]
    
    # (b) Find the position of the last non-zero coefficient
    last_nonzero_index = -1
    for i in range(len(ac_sequence) - 1, -1, -1):
        if ac_sequence[i] != 0:
            last_nonzero_index = i
            break
    
    # (c) Extract the sequence and add EOB
    # According to the previous verification, here we will keep the last '1' (corresponding to the matrix (3,2)), and then add EOB
    if last_nonzero_index == -1:
        symbols_to_encode = ['EOB']
    else:
        # Extract the value part
        symbols_to_encode = ac_sequence[:last_nonzero_index+1]
        # Convert the type to int to conform to the dict key
        symbols_to_encode = [int(x) for x in symbols_to_encode] 
        symbols_to_encode.append('EOB')

    print(f"Final encoding sequence (Verified): {symbols_to_encode}")
    # Expected output: [-3, 3, 0, 1, 0, 1, 2, 0, 2, 0, 0, -1, 0, 0, 0, 0, 0, 1, 'EOB']

    # Arithmetic encoding core calculation
    low = Decimal(0)
    high = Decimal(1)

    for sym in symbols_to_encode:
        range_val = high - low
        sym_low, sym_high = cdf[sym]
        
        # Update the interval: High must be based on the old Low, so first calculate the New High
        new_high = low + range_val * sym_high
        new_low = low + range_val * sym_low
        
        low = new_low
        high = new_high
    
    print(f"Final interval Low:  {low}")
    print(f"Final interval High: {high}")

    # Find the shortest binary string in the interval (Shortest Binary Code)
    # Target: find a binary decimal 0.b1b2b3... such that Low <= val < High
    
    output_bits = ""
    value = Decimal(0)
    power = Decimal(0.5) # 2^-1, 2^-2, ... (2^-1 = 0.5, 2^-2 = 0.25, ...)

    # Limit the loop times to prevent infinite loop (although theoretically it will converge)
    for _ in range(200): 
        mid = value + power
        
        if mid >= low and mid < high:
            # Case 1: after adding this bit (1), the value just falls within the interval
            # This is the shortest valid encoding, fill in 1 and end
            output_bits += "1"
            break
        elif mid < low:
            # Case 2: after adding this bit, the value is still less than the lower limit of the interval (Low)
            # So this bit must be 1 (make the value larger to catch up with the interval)，and continue to check the next bit
            output_bits += "1"
            value += power
        else:
            # Case 3: mid >= high
            # Adding this bit will exceed the upper limit of the interval, so this bit must be 0
            output_bits += "0"
            # value remains unchanged
            
        power /= 2

    print(f"Arithmetic encoding result (Binary String): {output_bits}")

    # 9. Save the result to A.txt
    with open("./results/A.txt", "w") as f:
        f.write(output_bits)

if __name__ == "__main__":
    main()