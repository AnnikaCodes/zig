// TODO

const Register = @import("bits.zig").Register;
const RegisterManagerFn = @import("../../register_manager.zig").RegisterManager;

// https://m680x0.github.io/doc/abi.html#calling-convention
pub const caller_preserved_regs = [_]Register{ .d0, .d1, .a0, .a1 };
pub const callee_preserved_regs = [_]Register{
    .d2, .d3, .d4, .d5, .d6, .d7,
    .a2, .a3, .a4, .a5,
};

const allocatable_registers = callee_preserved_regs ++ caller_preserved_regs;
pub const RegisterManager = RegisterManagerFn(@import("CodeGen.zig"), Register, &allocatable_registers);
