class BitPack:
    @staticmethod
    def pack_int4(to_pack: bytearray) -> bytearray:
        pack = bytearray()
        for i in range(0, len(to_pack), 2):
            pack.append((to_pack[i] & 0b1111) << 4 | to_pack[i + 1] & 0b1111)
        return pack

    @staticmethod
    def unpack_int4(to_unpack: bytearray) -> bytearray:
        unpack = bytearray()
        for i in to_unpack:
            high = (i & 0b11110000) >> 4
            if high & 0b1000 == 0b1000:
                high |= 0b11110000

            low = i & 0b1111
            if low & 0b1000 == 0b1000:
                low |= 0b11110000

            unpack.append(high)
            unpack.append(low)
        return unpack

    @staticmethod
    def print_int4(pack: bytearray) -> None:
        for i in pack:
            if i > 128:
                i -= 256
            print(i, end=", ")
        print()
