// TODO

const Emit = @This();
const std = @import("std");
const math = std.math;
const Mir = @import("Mir.zig");
const bits = @import("bits.zig");
const link = @import("../../link.zig");
const Module = @import("../../Module.zig");
const ErrorMsg = Module.ErrorMsg;
const assert = std.debug.assert;
const Instruction = bits.Instruction;
const Register = bits.Register;
const log = std.log.scoped(.m68k_emit);
const DebugInfoOutput = @import("../../codegen.zig").DebugInfoOutput;

mir: Mir,
bin_file: *link.File,
debug_output: DebugInfoOutput,
target: *const std.Target,
err_msg: ?*ErrorMsg = null,
src_loc: Module.SrcLoc,
code: *std.ArrayList(u8),

pub fn emitMir(emit: *Emit) !void {
    const mir_tags = emit.mir.instructions.items(.tag);

    // Emit machine code
    for (mir_tags, 0..) |tag, index| {
        const inst = @intCast(u32, index);
        switch (tag) {
            else => std.debug.print("TODO: emit MIR for {} (inst={}) on m68k\n", .{ tag, inst }),
        }
    }
}

pub fn deinit(emit: *Emit) void {
    emit.* = undefined;
}
