const std = @import("std");
const builtin = @import("builtin");
const mem = std.mem;
const math = std.math;
const assert = std.debug.assert;
const Air = @import("../../Air.zig");
const Mir = @import("Mir.zig");
const Emit = @import("Emit.zig");
const Liveness = @import("../../Liveness.zig");
const Type = @import("../../type.zig").Type;
const Value = @import("../../value.zig").Value;
const TypedValue = @import("../../TypedValue.zig");
const link = @import("../../link.zig");
const Module = @import("../../Module.zig");
const Compilation = @import("../../Compilation.zig");
const ErrorMsg = Module.ErrorMsg;
const Target = std.Target;
const Allocator = mem.Allocator;
const trace = @import("../../tracy.zig").trace;
const DW = std.dwarf;
const leb128 = std.leb;
const log = std.log.scoped(.codegen);
const build_options = @import("build_options");
const codegen = @import("../../codegen.zig");

const CodeGenError = codegen.CodeGenError;
const Result = codegen.Result;
const DebugInfoOutput = codegen.DebugInfoOutput;

const bits = @import("bits.zig");
const abi = @import("abi.zig");
const Register = bits.Register;
const RegisterManager = abi.RegisterManager;
const RegisterLock = RegisterManager.RegisterLock;
const Instruction = abi.Instruction;
const callee_preserved_regs = abi.callee_preserved_regs;
const gp = abi.RegisterClass.gp;

const InnerError = CodeGenError || error{OutOfRegisters};

gpa: Allocator,
air: Air,
liveness: Liveness,
bin_file: *link.File,
target: *const std.Target,
mod_fn: *const Module.Fn,
code: *std.ArrayList(u8),
debug_output: DebugInfoOutput,
err_msg: ?*ErrorMsg,
args: []MCValue,
ret_mcv: MCValue,
fn_type: Type,
arg_index: usize,
src_loc: Module.SrcLoc,
stack_align: u32,

/// MIR Instructions
mir_instructions: std.MultiArrayList(Mir.Inst) = .{},
/// MIR extra data
mir_extra: std.ArrayListUnmanaged(u32) = .{},

/// Byte offset within the source file of the ending curly.
end_di_line: u32,
end_di_column: u32,

/// The value is an offset into the `Function` `code` from the beginning.
/// To perform the reloc, write 32-bit signed little-endian integer
/// which is a relative jump, based on the address following the reloc.
exitlude_jump_relocs: std.ArrayListUnmanaged(usize) = .{},

/// Whenever there is a runtime branch, we push a Branch onto this stack,
/// and pop it off when the runtime branch joins. This provides an "overlay"
/// of the table of mappings from instructions to `MCValue` from within the branch.
/// This way we can modify the `MCValue` for an instruction in different ways
/// within different branches. Special consideration is needed when a branch
/// joins with its parent, to make sure all instructions have the same MCValue
/// across each runtime branch upon joining.
branch_stack: *std.ArrayList(Branch),

// Key is the block instruction
blocks: std.AutoHashMapUnmanaged(Air.Inst.Index, BlockData) = .{},

register_manager: RegisterManager = .{},
/// Maps offset to what is stored there.
stack: std.AutoHashMapUnmanaged(u32, StackAllocation) = .{},

/// Offset from the stack base, representing the end of the stack frame.
max_end_stack: u32 = 0,
/// Represents the current end stack offset. If there is no existing slot
/// to place a new stack allocation, it goes here, and then bumps `max_end_stack`.
next_stack_offset: u32 = 0,

/// Debug field, used to find bugs in the compiler.
air_bookkeeping: @TypeOf(air_bookkeeping_init) = air_bookkeeping_init,

const air_bookkeeping_init = if (std.debug.runtime_safety) @as(usize, 0) else {};

const MCValue = union(enum) {
    /// No runtime bits. `void` types, empty structs, u0, enums with 1 tag, etc.
    /// TODO Look into deleting this tag and using `dead` instead, since every use
    /// of MCValue.none should be instead looking at the type and noticing it is 0 bits.
    none,
    /// Control flow will not allow this value to be observed.
    unreach,
    /// No more references to this value remain.
    dead,
    /// The value is undefined.
    undef,
    /// A pointer-sized integer that fits in a register.
    /// If the type is a pointer, this is the pointer address in virtual address space.
    immediate: u64,
    /// The value is in a target-specific register.
    register: Register,
    /// The value is in memory at a hard-coded address.
    /// If the type is a pointer, it means the pointer address is at this memory location.
    memory: u64,
    /// The value is one of the stack variables.
    /// If the type is a pointer, it means the pointer address is in the stack at this offset.
    stack_offset: u32,
    /// The value is a pointer to one of the stack variables (payload is stack offset).
    ptr_stack_offset: u32,
};

const Branch = struct {
    inst_table: std.AutoArrayHashMapUnmanaged(Air.Inst.Index, MCValue) = .{},

    fn deinit(self: *Branch, gpa: Allocator) void {
        self.inst_table.deinit(gpa);
        self.* = undefined;
    }
};

const StackAllocation = struct {
    inst: Air.Inst.Index,
    /// TODO do we need size? should be determined by inst.ty.abiSize()
    size: u32,
};

const BlockData = struct {
    relocs: std.ArrayListUnmanaged(Mir.Inst.Index),
    /// The first break instruction encounters `null` here and chooses a
    /// machine code value for the block result, populating this field.
    /// Following break instructions encounter that value and use it for
    /// the location to store their block results.
    mcv: MCValue,
};

const BigTomb = struct {
    function: *Self,
    inst: Air.Inst.Index,
    lbt: Liveness.BigTomb,

    fn feed(bt: *BigTomb, op_ref: Air.Inst.Ref) void {
        const dies = bt.lbt.feed();
        const op_index = Air.refToIndex(op_ref) orelse return;
        if (!dies) return;
        bt.function.processDeath(op_index);
    }

    fn finishAir(bt: *BigTomb, result: MCValue) void {
        const is_used = !bt.function.liveness.isUnused(bt.inst);
        if (is_used) {
            log.debug("%{d} => {}", .{ bt.inst, result });
            const branch = &bt.function.branch_stack.items[bt.function.branch_stack.items.len - 1];
            branch.inst_table.putAssumeCapacityNoClobber(bt.inst, result);
        }
        bt.function.finishAirBookkeeping();
    }
};

const Self = @This();

pub fn generate(
    bin_file: *link.File,
    src_loc: Module.SrcLoc,
    module_fn: *Module.Fn,
    air: Air,
    liveness: Liveness,
    code: *std.ArrayList(u8),
    debug_output: DebugInfoOutput,
    module: *Module,
) CodeGenError!Result {
    if (build_options.skip_non_native and builtin.cpu.arch != bin_file.options.target.cpu.arch) {
        @panic("Attempted to compile for architecture that was disabled by build configuration");
    }

    const mod = bin_file.options.module.?;
    const fn_owner_decl = mod.declPtr(module_fn.owner_decl);
    assert(fn_owner_decl.has_tv);
    const fn_type = fn_owner_decl.ty;

    var branch_stack = std.ArrayList(Branch).init(bin_file.allocator);
    defer {
        assert(branch_stack.items.len == 1);
        branch_stack.items[0].deinit(bin_file.allocator);
        branch_stack.deinit();
    }
    try branch_stack.append(.{});

    var function = Self{
        .gpa = bin_file.allocator,
        .air = air,
        .liveness = liveness,
        .target = &bin_file.options.target,
        .bin_file = bin_file,
        .mod_fn = module_fn,
        .code = code,
        .debug_output = debug_output,
        .err_msg = null,
        .args = undefined, // populated after `resolveCallingConventionValues`
        .ret_mcv = undefined, // populated after `resolveCallingConventionValues`
        .fn_type = fn_type,
        .arg_index = 0,
        .branch_stack = &branch_stack,
        .src_loc = src_loc,
        .stack_align = undefined,
        .end_di_line = module_fn.rbrace_line,
        .end_di_column = module_fn.rbrace_column,
    };
    defer function.stack.deinit(bin_file.allocator);
    defer function.blocks.deinit(bin_file.allocator);
    defer function.exitlude_jump_relocs.deinit(bin_file.allocator);

    var call_info = function.resolveCallingConventionValues(fn_type, module) catch |err| return err;
    defer call_info.deinit(&function);

    function.args = call_info.args;
    function.ret_mcv = call_info.return_value;
    function.stack_align = call_info.stack_align;
    function.max_end_stack = call_info.stack_byte_count;

    function.gen() catch |err| switch (err) {
        error.CodegenFail => return Result{ .fail = function.err_msg.? },
        error.OutOfRegisters => return Result{
            .fail = try ErrorMsg.create(bin_file.allocator, src_loc, "CodeGen ran out of registers. This is a bug in the Zig compiler.", .{}),
        },
        else => |e| return e,
    };

    var mir = Mir{
        .instructions = function.mir_instructions.toOwnedSlice(),
        .extra = try function.mir_extra.toOwnedSlice(bin_file.allocator),
    };
    defer mir.deinit(bin_file.allocator);

    var emit = Emit{
        .mir = mir,
        .bin_file = bin_file,
        .debug_output = debug_output,
        .target = &bin_file.options.target,
        .src_loc = src_loc,
        .code = code,
    };
    defer emit.deinit();

    emit.emitMir() catch |err| switch (err) {
        error.EmitFail => return Result{ .fail = emit.err_msg.? },
        else => |e| return e,
    };

    if (function.err_msg) |em| {
        return Result{ .fail = em };
    } else {
        return Result.ok;
    }
}

fn addInst(self: *Self, inst: Mir.Inst) error{OutOfMemory}!Mir.Inst.Index {
    const gpa = self.gpa;

    try self.mir_instructions.ensureUnusedCapacity(gpa, 1);

    const result_index = @intCast(Air.Inst.Index, self.mir_instructions.len);
    self.mir_instructions.appendAssumeCapacity(inst);
    return result_index;
}

pub fn addExtra(self: *Self, extra: anytype) Allocator.Error!u32 {
    const fields = std.meta.fields(@TypeOf(extra));
    try self.mir_extra.ensureUnusedCapacity(self.gpa, fields.len);
    return self.addExtraAssumeCapacity(extra);
}

pub fn addExtraAssumeCapacity(self: *Self, extra: anytype) u32 {
    const fields = std.meta.fields(@TypeOf(extra));
    const result = @intCast(u32, self.mir_extra.items.len);
    inline for (fields) |field| {
        self.mir_extra.appendAssumeCapacity(switch (field.type) {
            u32 => @field(extra, field.name),
            i32 => @bitCast(u32, @field(extra, field.name)),
            else => @compileError("bad field type"),
        });
    }
    return result;
}

fn gen(self: *Self) !void {
    _ = self;
    std.debug.panic("TODO: implement m68k CodeGen.gen", .{});
}

fn genBody(self: *Self, body: []const Air.Inst.Index) InnerError!void {
    const air_tags = self.air.instructions.items(.tag);

    for (body) |inst| {
        const old_air_bookkeeping = self.air_bookkeeping;
        try self.ensureProcessDeathCapacity(Liveness.bpi);

        switch (air_tags[inst]) {
            // zig fmt: off
            _ => std.debug.panic("AIR tag is not mplemented for m68k: {}", .{air_tags[inst]})
            // zig fmt: on
        }
        if (std.debug.runtime_safety) {
            if (self.air_bookkeeping < old_air_bookkeeping + 1) {
                std.debug.panic("in CodeGen.zig, handling of AIR instruction %{d} ('{}') did not do proper bookkeeping. Look for a missing call to finishAir.", .{ inst, air_tags[inst] });
            }
        }
    }
}

/// Asserts there is already capacity to insert into top branch inst_table.
fn processDeath(self: *Self, inst: Air.Inst.Index) void {
    const air_tags = self.air.instructions.items(.tag);
    if (air_tags[inst] == .constant) return; // Constants are immortal.
    // When editing this function, note that the logic must synchronize with `reuseOperand`.
    const prev_value = self.getResolvedInstValue(inst);
    const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
    branch.inst_table.putAssumeCapacity(inst, .dead);
    switch (prev_value) {
        .register => |reg| {
            self.register_manager.freeReg(reg);
        },
        else => {}, // TODO process stack allocation death
    }
}

/// Called when there are no operands, and the instruction is always unreferenced.
fn finishAirBookkeeping(self: *Self) void {
    if (std.debug.runtime_safety) {
        self.air_bookkeeping += 1;
    }
}

fn finishAir(self: *Self, inst: Air.Inst.Index, result: MCValue, operands: [Liveness.bpi - 1]Air.Inst.Ref) void {
    var tomb_bits = self.liveness.getTombBits(inst);
    for (operands) |op| {
        const dies = @truncate(u1, tomb_bits) != 0;
        tomb_bits >>= 1;
        if (!dies) continue;
        const op_int = @enumToInt(op);
        if (op_int < Air.Inst.Ref.typed_value_map.len) continue;
        const op_index = @intCast(Air.Inst.Index, op_int - Air.Inst.Ref.typed_value_map.len);
        self.processDeath(op_index);
    }
    const is_used = @truncate(u1, tomb_bits) == 0;
    if (is_used) {
        log.debug("%{d} => {}", .{ inst, result });
        const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
        branch.inst_table.putAssumeCapacityNoClobber(inst, result);

        switch (result) {
            .register => |reg| {
                // In some cases (such as bitcast), an operand
                // may be the same MCValue as the result. If
                // that operand died and was a register, it
                // was freed by processDeath. We have to
                // "re-allocate" the register.
                if (self.register_manager.isRegFree(reg)) {
                    self.register_manager.getRegAssumeFree(reg, inst);
                }
            },
            else => {},
        }
    }
    self.finishAirBookkeeping();
}

fn ensureProcessDeathCapacity(self: *Self, additional_count: usize) !void {
    const table = &self.branch_stack.items[self.branch_stack.items.len - 1].inst_table;
    try table.ensureUnusedCapacity(self.gpa, additional_count);
}

const CallMCValues = struct {
    args: []MCValue,
    return_value: MCValue,
    stack_byte_count: u32,
    stack_align: u32,

    fn deinit(self: *CallMCValues, func: *Self) void {
        func.gpa.free(self.args);
        self.* = undefined;
    }
};

// Via Zig 6502:
/// This tells us where the arguments and the return value should be based on a function's prototype.
/// This function must always give the same output for the same input.
///
/// The reason we need calling conventions is that a function code is included only once
/// in the binary and it can thus only take in the arguments in one specific way
/// and so all call sites need to do it the same way for it to work out.
///
/// So, this is called both when we call a function and once that function is generated.
/// In both cases for the same function we will receive the same call values.
/// Caller must call `CallMCValues.deinit`.
fn resolveCallingConventionValues(self: *Self, fn_ty: Type, module: *Module) !CallMCValues {
    const cc = fn_ty.fnCallingConvention();
    const param_types = try self.gpa.alloc(Type, fn_ty.fnParamLen());
    defer self.gpa.free(param_types);
    fn_ty.fnParamTypes(param_types);
    var result: CallMCValues = .{
        .args = try self.gpa.alloc(MCValue, param_types.len),
        // These undefined values must be populated before returning from this function.
        .return_value = undefined,
        .stack_byte_count = undefined,
        .stack_align = undefined,
    };
    errdefer self.gpa.free(result.args);

    const ret_ty = fn_ty.fnReturnType();

    // According to https://m680x0.github.io/doc/abi.html#calling-convention,
    // we just pass all paramaters on the stack
    switch (cc) {
        .Unspecified, .C => {
            if (cc == .C) {
                std.debug.print(
                    "warning: the m68k backend's C calling convention currently uses the System V C ABI!\n" ++
                    "This is likely incompatible with your modern C compiler, and decisions have been made " ++
                    "about how to handle modern types with which you, or your compiler, may disagree!\n", .{}
                );
            }

            var stack_offset: u32 = 0;

            if (ret_ty.zigTypeTag() == .NoReturn) {
                result.return_value = .{ .unreach = {} };
            } else if (!ret_ty.hasRuntimeBitsIgnoreComptime()) {
                result.return_value = .{ .none = {} };
            } else {
                // Per https://m680x0.github.io/doc/abi.html#calling-convention,
                // return values go in %d0 if 'integral' or otherwise are passed by memory with the address
                // in %a0 (SysV ABI, maybe gcc is different — it's not too clear)
                //
                // Probably a good TODO to read the GCC source and figure out compatibility.
                // Can also consider developing a Zig custom ABI for m68k.

                const ret_ty_bits = ret_ty.abiSize(self.target.*) * 8;

                // OK, so what the heck makes a return value 'integral'?
                // Well, the term is referenced in a 1990 book about the SystemV m68k ABI,
                // https://media.githubusercontent.com/media/M680x0/Literature/master/sysv-m68k-abi.pdf
                // (TODO: maybe put copies these various docs I keep referencing on my website in case links break)
                //
                // Anyway, old SysV book provides a list of integral types on page 3-2, figure 3-1:
                // * signed char
                // * char
                // * short
                // * unsigned short
                // * int
                // * signed int
                // * long
                // * signed long
                // * enum
                // * unsigned int
                // * unsigned long
                //
                // Basically, it's any scalar (non-aggregate type) that isn't a pointer or a floating-point number.
                // Of course, over three decades later,
                // we have now invented the concept of numbers that don't fit in CPU registers.
                // So we'll just pass return values that are larger than 32 bits by memory, and follow SysV otherwise.
                //
                // TODO: it's probably possible to optimize zero size types here since Zig has those and SysV doesn't.
                if (ret_ty_bits <= 32) { // registers are 32 bits on m68k
                    result.return_value = .{ .register = .d0 };
                } else if (ret_ty.zigTypeTag() == .Pointer) {
                    // Per page 3-16 of the System V ABI manual, functions that return pointers use %a0,
                    // the first address register.
                    result.return_value = .{ .register = .a0 };
                } else {
                    // Aggregate types and scalar types that don't fit in %d0 *should* be passed by memory.
                    // The address of the return value is passed in %a0, according to manual page 3-17.
                    //
                    // However, I'm not sure how to do this in this function without using the stack, since
                    // we don't know the address yet.
                    //
                    // Maybe we need to panic only if it's the SysV ABI and do something novel for the Zig ABI?
                    std.debug.print(
                        "TODO: m68k backend: return values ({}, {}) that don't fit in a register " ++
                        "(try using a pointer)\n",
                        .{ret_ty.fmt(module), ret_ty.zigTypeTag()} // fmtDebug() doesn't work due to a TODO in src/type.zig
                    );
                    stack_offset += 4; // ptr width
                }
            }

            // Parameters are passed on the stack, according to the System V ABI manual page 3-12.
            for (param_types, 0..) |ty, i| {
                if (ty.abiSize(self.target.*) > 0) {
                    // i copied this from aarch64 and x86_64 please dont blow up in my face
                    const param_size = @intCast(u32, ty.abiSize(self.target.*));
                    const param_alignment = ty.abiAlignment(self.target.*);

                    const offset = std.mem.alignForwardGeneric(u32, stack_offset + param_size, param_alignment);
                    result.args[i] = .{ .stack_offset = offset };
                    stack_offset = offset;
                } else {
                    result.args[i] = .{ .none = {} };
                }
            }
        },
        else => std.debug.panic("TODO: m68k backend: implement resolveCallingConventionValues for ABI '{}'", .{cc}),
    }

    return result;
}
