export fn entry() u32 {
    var bytes: [4]u8 = [_]u8{0x01, 0x02, 0x03, 0x04};
    const ptr = @ptrCast(*u32, &bytes[0]);
    return ptr.*;
}

// increase pointer alignment in @ptrCast
//
// tmp.zig:3:17: error: cast increases pointer alignment
// tmp.zig:3:38: note: '*u8' has alignment 1
// tmp.zig:3:26: note: '*u32' has alignment 4
