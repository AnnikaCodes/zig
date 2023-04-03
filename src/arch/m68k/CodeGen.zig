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
    immediate: u32,
    /// The value is in a target-specific register.
    register: Register,
    /// The value is in memory at a hard-coded address.
    /// If the type is a pointer, it means the pointer address is at this memory location.
    memory: u32,
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


/// TODO support scope overrides. Also note this logic is duplicated with `Module.wantSafety`.
fn wantSafety(self: *Self) bool {
    return switch (self.bin_file.options.optimize_mode) {
        .Debug => true,
        .ReleaseSafe => true,
        .ReleaseFast => false,
        .ReleaseSmall => false,
    };
}

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
    const cc = self.fn_type.fnCallingConvention();
    // who even knows

    // ok so i think this is something called a Function Prologue and Epilogue,
    // https://en.wikipedia.org/wiki/Function_prologue_and_epilogue

    // according to GCC, this is how you do a function in m68k:
    // * linkw %fp,#0
    // * nop (is this necessary? maybe it's just part of the fn body)
    //
    // * <fn body>
    //
    // * unlk %fp
    // * rts

    // but sysV has a different prologue according to
    // https://media.githubusercontent.com/media/M680x0/Literature/master/sysv-m68k-abi.pdf
    // it's:
    // * linkl %fp,&-80
    // * movem.l %d7/%a5, -(%sp)
    // * fmovm.x %fp2, -(%sp) (floating point oly, we dont need)
    //
    // * <function body>
    //
    // * mov.l %a5, %a0
    // * fmovm.x (%sp+) (floating point only, we dont need)
    // * movm.l (%sp+), %d7/%a5
    // * unlk %fp
    // * rts
    // (note that the SysV manual says `movm` while the Motorola manual says `movem`)

    // for now i'm doing the one GCC generates but... if that breaks we can try the SysV one
    if (cc != .Naked) {
        // linkw %fp,#0
        _ = try self.addInst(.{
            .tag = .LINK,
            .data = .{
                .register_and_displacement = .{
                    .register = .a6, // a6 = fp
                    .displacement = 0,
                },
            }
        });

        // nop
        _ = try self.addInst(.{
            .tag = .NOP,
            .data = .{ .none = {} },
        });

        try self.genBody(self.air.getMainBody());

        // unlk %fp
        _ = try self.addInst(.{
            .tag = .UNLK,
            .data = .{
                .register = .a6, // a6 = fp
            },
        });

        // rts
        _ = try self.addInst(.{
            .tag = .RTS,
            .data = .{ .none = {} },
        });
    } else {
        try self.genBody(self.air.getMainBody());
    }
}


fn performReloc(self: *Self, inst: Mir.Inst.Index) !void {
    const tag = self.mir_instructions.items(.tag)[inst];
    switch (tag) {
        // TODO: does this even work?
        .JMP => {
            self.mir_instructions.items(.data)[inst].inst = @intCast(Mir.Inst.Index, self.mir_instructions.len);
        },
        else => std.debug.panic("TODO performReloc for tag {}", .{tag}),
    }
}


fn genBody(self: *Self, body: []const Air.Inst.Index) InnerError!void {
    const air_tags = self.air.instructions.items(.tag);

    for (body) |inst| {
        const old_air_bookkeeping = self.air_bookkeeping;
        try self.ensureProcessDeathCapacity(Liveness.bpi);

        switch (air_tags[inst]) {
            .alloc              => try self.airAlloc(inst),
            .arg                => try self.airArg(inst),
            .bitcast            => try self.airBitCast(inst),
            .block              => try self.airBlock(inst),

            .dbg_block_begin,
            .dbg_block_end      => try self.airDbgBlock(inst),

            .dbg_stmt           => try self.airDbgStmt(inst),

            .dbg_var_ptr,
            .dbg_var_val        => try self.airDbgVar(inst),

            .load               => try self.airLoad(inst),
            .loop               => try self.airLoop(inst),
            .store              => try self.airStore(inst),
            .slice_len          => try self.airSliceLen(inst),

            else => std.debug.panic(
                "m68k backend: lowering for AIR tag {} is not yet implemented",
                .{air_tags[inst]},
            ),
        }
        if (std.debug.runtime_safety) {
            if (self.air_bookkeeping < old_air_bookkeeping + 1) {
                std.debug.panic("in CodeGen.zig, handling of AIR instruction %{d} ('{}') did not do proper bookkeeping. Look for a missing call to finishAir.", .{ inst, air_tags[inst] });
            }
        }
    }
}


fn getResolvedInstValue(self: *Self, inst: Air.Inst.Index) MCValue {
    // Treat each stack item as a "layer" on top of the previous one.
    var i: usize = self.branch_stack.items.len;
    while (true) {
        i -= 1;
        if (self.branch_stack.items[i].inst_table.get(inst)) |mcv| {
            assert(mcv != .dead);
            return mcv;
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

fn airAlloc(self: *Self, inst: Air.Inst.Index) !void {
    const stack_offset = try self.allocMemPtr(inst);
    return self.finishAir(inst, .{ .ptr_stack_offset = stack_offset }, .{ .none, .none, .none });
}

fn airArg(self: *Self, inst: Air.Inst.Index) !void {
    // stolen from ARM pls no break
    // skip zero-bit arguments as they don't have a corresponding arg instruction
    var arg_index = self.arg_index;
    while (self.args[arg_index] == .none) arg_index += 1;
    self.arg_index = arg_index + 1;

    const result: MCValue = if (self.liveness.isUnused(inst)) .dead else self.args[arg_index];
    return self.finishAir(inst, result, .{ .none, .none, .none });
}

fn airBitCast(self: *Self, inst: Air.Inst.Index) !void {
    const ty_op = self.air.instructions.items(.data)[inst].ty_op;
    const result = if (self.liveness.isUnused(inst)) .dead else result: {
        const operand = try self.resolveInst(ty_op.operand);
        if (self.reuseOperand(inst, ty_op.operand, 0, operand)) break :result operand;

        const operand_lock = switch (operand) {
            .register => |reg| self.register_manager.lockReg(reg),
            else => other: {
                std.debug.print("m68k backend: warning: don't know how to lock operand {}\n", .{operand});
                break :other null;
            },
        };
        defer if (operand_lock) |lock| self.register_manager.unlockReg(lock);

        const dest_ty = self.air.typeOfIndex(inst);
        const dest = try self.allocRegOrMem(inst, true);
        try self.setRegOrMem(dest_ty, dest, operand);
        break :result dest;
    };
    return self.finishAir(inst, result, .{ ty_op.operand, .none, .none });
}


// copied from aarch64
fn airBlock(self: *Self, inst: Air.Inst.Index) !void {
    try self.blocks.putNoClobber(self.gpa, inst, .{
        // A block is a setup to be able to jump to the end.
        .relocs = .{},
        // It also acts as a receptacle for break operands.
        // Here we use `MCValue.none` to represent a null value so that the first
        // break instruction will choose a MCValue for the block result and overwrite
        // this field. Following break instructions will use that MCValue to put their
        // block results.
        .mcv = MCValue{ .none = {} },
    });
    defer self.blocks.getPtr(inst).?.relocs.deinit(self.gpa);

    const ty_pl = self.air.instructions.items(.data)[inst].ty_pl;
    const extra = self.air.extraData(Air.Block, ty_pl.payload);
    const body = self.air.extra[extra.end..][0..extra.data.body_len];
    try self.genBody(body);

    // relocations for `br` instructions
    const relocs = &self.blocks.getPtr(inst).?.relocs;
    if (relocs.items.len > 0 and relocs.items[relocs.items.len - 1] == self.mir_instructions.len - 1) {
        // If the last Mir instruction is the last relocation (which
        // would just jump one instruction further), it can be safely
        // removed
        self.mir_instructions.orderedRemove(relocs.pop());
    }
    for (relocs.items) |reloc| {
        try self.performReloc(reloc);
    }

    const result = self.blocks.getPtr(inst).?.mcv;
    return self.finishAir(inst, result, .{ .none, .none, .none });
}

fn airDbgBlock(self: *Self, inst: Air.Inst.Index) !void {
    // TODO emit debug info lexical block
    return self.finishAir(inst, .dead, .{ .none, .none, .none });
}

fn airDbgStmt(self: *Self, inst: Air.Inst.Index) !void {
    const dbg_stmt = self.air.instructions.items(.data)[inst].dbg_stmt;

    _ = try self.addInst(.{
        .tag = .dbg_line,
        .data = .{
            .dbg_line_column = .{
                .line = dbg_stmt.line,
                .column = dbg_stmt.column,
            },
        },
    });

    return self.finishAirBookkeeping();
}

fn airDbgVar(self: *Self, inst: Air.Inst.Index) !void {
    const pl_op = self.air.instructions.items(.data)[inst].pl_op;
    const name = self.air.nullTerminatedString(pl_op.payload);
    const operand = pl_op.operand;
    // TODO emit debug info for this variable
    _ = name;
    return self.finishAir(inst, .dead, .{ operand, .none, .none });
}

/// Load a value from a pointer
fn airLoad(self: *Self, inst: Air.Inst.Index) !void {
    const ty_op = self.air.instructions.items(.data)[inst].ty_op;
    const elem_ty = self.air.typeOfIndex(inst);
    const result: MCValue = result: {
        if (!elem_ty.hasRuntimeBits())
            break :result MCValue.none;

        const ptr = try self.resolveInst(ty_op.operand);
        const is_volatile = self.air.typeOf(ty_op.operand).isVolatilePtr();
        if (self.liveness.isUnused(inst) and !is_volatile)
            break :result MCValue.dead;

        const dst_mcv: MCValue = blk: {
            if (self.reuseOperand(inst, ty_op.operand, 0, ptr)) {
                // The MCValue that holds the pointer can be re-used as the value.
                break :blk ptr;
            } else {
                break :blk try self.allocRegOrMem(inst, true);
            }
        };
        try self.load(dst_mcv, ptr, self.air.typeOf(ty_op.operand));
        break :result dst_mcv;
    };
    return self.finishAir(inst, result, .{ ty_op.operand, .none, .none });
}

fn airLoop(self: *Self, inst: Air.Inst.Index) !void {
    // A loop is a setup to be able to jump back to the beginning.
    const ty_pl = self.air.instructions.items(.data)[inst].ty_pl;
    const loop = self.air.extraData(Air.Block, ty_pl.payload);
    const body = self.air.extra[loop.end..][0..loop.data.body_len];
    const start_index = @intCast(Mir.Inst.Index, self.mir_instructions.len);
    try self.genBody(body);
    try self.jump(start_index);
    return self.finishAirBookkeeping();
}

fn airStore(self: *Self, inst: Air.Inst.Index) !void {
    const bin_op = self.air.instructions.items(.data)[inst].bin_op;
    const ptr = try self.resolveInst(bin_op.lhs);
    const value = try self.resolveInst(bin_op.rhs);
    const ptr_ty = self.air.typeOf(bin_op.lhs);
    const value_ty = self.air.typeOf(bin_op.rhs);

    try self.store(ptr, value, ptr_ty, value_ty);

    return self.finishAir(inst, .dead, .{ bin_op.lhs, bin_op.rhs, .none });
}

fn airSliceLen(self: *Self, inst: Air.Inst.Index) !void {
    const ty_op = self.air.instructions.items(.data)[inst].ty_op;
    const result: MCValue = if (self.liveness.isUnused(inst)) .dead else result: {
        const mcv = try self.resolveInst(ty_op.operand);
        switch (mcv) {
            .dead, .unreach, .none => unreachable,
            .stack_offset => |off| {
                break :result MCValue{ .stack_offset = off - 4 };
            },
            .memory => |addr| {
                break :result MCValue{ .memory = addr + 4 };
            },
            else => std.debug.panic("TODO: m68k backend: implement slice_len for {}", .{mcv}),
        }
    };
    return self.finishAir(inst, result, .{ ty_op.operand, .none, .none });
}

fn store(self: *Self, ptr: MCValue, value: MCValue, ptr_ty: Type, value_ty: Type) !void {
    const abi_size = @intCast(u32, value_ty.abiSize(self.target.*));

    switch (ptr) {
        .none => unreachable,
        .undef => unreachable,
        .unreach => unreachable,
        .dead => unreachable,
        .immediate => |imm| try self.setRegOrMem(value_ty, .{ .memory = imm }, value),
        .register => |reg| {
            switch (value) {
                .none, .unreach, .dead => unreachable,
                .undef => {
                    if (self.wantSafety()) {
                        switch (abi_size) {
                            1 => try self.store(ptr, .{ .immediate = 0xAA }, ptr_ty, value_ty),
                            2 => try self.store(ptr, .{ .immediate = 0xAAAA }, ptr_ty, value_ty),
                            4 => try self.store(ptr, .{ .immediate = 0xAAAAAAAA }, ptr_ty, value_ty),
                            else => std.debug.panic("m68k backend: don't know how to store undefined of size {} to register {}", .{ abi_size, reg }),
                        }
                    }
                },
                .register => |src_reg| {
                    // MOVE src_reg, reg
                    _ = try self.addInst(.{
                        .tag = .MOVE,
                        .data = .{
                            .src_dest = .{
                                .src = .{ .register = src_reg },
                                .dest = .{ .register = reg },
                            }
                        }
                    });
                },
                .immediate => |imm| try self.genSetReg(value_ty, reg, .{ .immediate = imm }),
                .memory => |addr| try self.genSetReg(value_ty, reg, .{ .memory = addr }),
                else => std.debug.panic("m68k backend: don't know how to store {} to register {}", .{ value, reg }),
            }
        },
        .ptr_stack_offset => |off| try self.genSetStack(value_ty, off, value),
        else => std.debug.panic("TODO: m68k backend: implement storing to {}", .{ptr}),
    }
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
                        "warning: m68k backend: return values (like the {} here) that don't fit in a register " ++
                        "might not work properly\n",
                        .{ret_ty.fmt(module)} // fmtDebug() doesn't work due to a TODO in src/type.zig
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

fn allocMem(self: *Self, inst: Air.Inst.Index, abi_size: u32, abi_align: u32) !u32 {
    if (abi_align > self.stack_align)
        self.stack_align = abi_align;
    // TODO find a free slot instead of always appending
    const offset = mem.alignForwardGeneric(u32, self.next_stack_offset + abi_size, abi_align);
    self.next_stack_offset = offset;
    if (self.next_stack_offset > self.max_end_stack)
        self.max_end_stack = self.next_stack_offset;
    try self.stack.putNoClobber(self.gpa, offset, .{
        .inst = inst,
        .size = abi_size,
    });
    return offset;
}

// Use a pointer instruction as the basis for allocating stack memory.
fn allocMemPtr(self: *Self, inst: Air.Inst.Index) !u32 {
    const ptr_ty = self.air.typeOfIndex(inst);
    const elem_ty = ptr_ty.elemType();

    if (!elem_ty.hasRuntimeBitsIgnoreComptime()) {
        return self.allocMem(inst, @sizeOf(usize), @alignOf(usize));
    }

    const abi_size = math.cast(u32, elem_ty.abiSize(self.target.*)) orelse {
        const mod = self.bin_file.options.module.?;
        return std.debug.panic(
            "m68k backend: type '{}' too big to fit into stack frame",
            .{elem_ty.fmt(mod)},
        );
    };
    // TODO swap this for inst.ty.ptrAlign
    const abi_align = ptr_ty.ptrAlignment(self.target.*);
    return self.allocMem(inst, abi_size, abi_align);
}

fn allocRegOrMem(self: *Self, inst: Air.Inst.Index, reg_ok: bool) !MCValue {
    const elem_ty = self.air.typeOfIndex(inst);
    const abi_size = math.cast(u32, elem_ty.abiSize(self.target.*)) orelse {
        const mod = self.bin_file.options.module.?;
        return std.debug.panic("m68k: type '{}' too big to fit into stack frame", .{elem_ty.fmt(mod)});
    };
    const abi_align = elem_ty.abiAlignment(self.target.*);
    if (abi_align > self.stack_align)
        self.stack_align = abi_align;

    if (reg_ok) {
        // Make sure the type can fit in a register before we try to allocate one.
        const ptr_bits = self.target.cpu.arch.ptrBitWidth();
        const ptr_bytes: u64 = @divExact(ptr_bits, 8);
        if (abi_size <= ptr_bytes) {
            if (self.register_manager.tryAllocReg(inst, gp)) |reg| {
                return MCValue{ .register = reg };
            }
        }
    }
    const stack_offset = try self.allocMem(inst, abi_size, abi_align);
    return MCValue{ .stack_offset = stack_offset };
}


fn genTypedValue(self: *Self, typed_value: TypedValue) !MCValue {
    const mcv: MCValue = switch (try codegen.genTypedValue(
        self.bin_file,
        self.src_loc,
        typed_value,
        self.mod_fn.owner_decl,
    )) {
        .mcv => |mcv| switch (mcv) {
            .none => .none,
            .undef => .undef,
            .linker_load => unreachable, // TODO
            .immediate => |imm| {
                if (imm >= std.math.pow(u64, 2, 32)) {
                    std.debug.panic("m68k backend: immediate value {} is greater than 32 bits!", .{imm});
                }
                return .{ .immediate = @intCast(u32, imm) };
            },
            .memory => |addr| return .{ .memory = @intCast(u32, addr) },
        },
        .fail => |msg| {
            // self.err_msg = msg;
            // return error.CodegenFail;
            std.debug.panic("m68k backend: couldn't genTypedValue: {}", .{msg});
        },
    };
    return mcv;
}


fn resolveInst(self: *Self, inst: Air.Inst.Ref) !MCValue {
    // First section of indexes correspond to a set number of constant values.
    const ref_int = @enumToInt(inst);
    if (ref_int < Air.Inst.Ref.typed_value_map.len) {
        const tv = Air.Inst.Ref.typed_value_map[ref_int];
        if (!tv.ty.hasRuntimeBits()) {
            return MCValue{ .none = {} };
        }
        return self.genTypedValue(tv);
    }

    // If the type has no codegen bits, no need to store it.
    const inst_ty = self.air.typeOf(inst);
    if (!inst_ty.hasRuntimeBits()) {
        return MCValue{ .none = {} };
    }

    const inst_index = @intCast(Air.Inst.Index, ref_int - Air.Inst.Ref.typed_value_map.len);
    switch (self.air.instructions.items(.tag)[inst_index]) {
        .constant => {
            // Constants have static lifetimes, so they are always memoized in the outer most table.
            const branch = &self.branch_stack.items[0];
            const gop = try branch.inst_table.getOrPut(self.gpa, inst_index);
            if (!gop.found_existing) {
                const ty_pl = self.air.instructions.items(.data)[inst_index].ty_pl;
                gop.value_ptr.* = try self.genTypedValue(.{
                    .ty = inst_ty,
                    .val = self.air.values[ty_pl.payload],
                });
            }
            return gop.value_ptr.*;
        },
        .const_ty => unreachable,
        else => return self.getResolvedInstValue(inst_index),
    }
}

fn reuseOperand(self: *Self, inst: Air.Inst.Index, operand: Air.Inst.Ref, op_index: Liveness.OperandInt, mcv: MCValue) bool {
    if (!self.liveness.operandDies(inst, op_index))
        return false;

    switch (mcv) {
        .register => |reg| {
            // If it's in the registers table, need to associate the register with the
            // new instruction.
            if (RegisterManager.indexOfRegIntoTracked(reg)) |index| {
                if (!self.register_manager.isRegFree(reg)) {
                    self.register_manager.registers[index] = inst;
                }
            }
            log.debug("%{d} => {} (reused)", .{ inst, reg });
        },
        .stack_offset => |off| {
            log.debug("%{d} => stack offset {d} (reused)", .{ inst, off });
        },
        else => return false,
    }

    // Prevent the operand deaths processing code from deallocating it.
    self.liveness.clearOperandDeath(inst, op_index);

    // That makes us responsible for doing the rest of the stuff that processDeath would have done.
    const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
    branch.inst_table.putAssumeCapacity(Air.refToIndex(operand).?, .dead);

    return true;
}

/// Sets the value without any modifications to register allocation metadata or stack allocation metadata.
fn setRegOrMem(self: *Self, ty: Type, loc: MCValue, val: MCValue) !void {
    switch (loc) {
        .none => return,
        .register => |reg| return self.genSetReg(ty, reg, val),
        .stack_offset => |off| return self.genSetStack(ty, off, val),
        .memory => std.debug.panic("TODO implement setRegOrMem for memory", .{}),
        else => unreachable,
    }
}

pub fn spillInstruction(self: *Self, reg: Register, inst: Air.Inst.Index) !void {
    const stack_mcv = try self.allocRegOrMem(inst, false);
    log.debug("spilling {d} to stack mcv {any}", .{ inst, stack_mcv });
    const reg_mcv = self.getResolvedInstValue(inst);
    assert(reg == reg_mcv.register);
    const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
    try branch.inst_table.put(self.gpa, inst, stack_mcv);
    try self.genSetStack(self.air.typeOfIndex(inst), stack_mcv.stack_offset, reg_mcv);
}


fn genSetStack(self: *Self, ty: Type, stack_offset: u32, mcv: MCValue) !void {
    switch (mcv) {
        .dead => unreachable,
        .unreach, .none => return, // Nothing to do.
        .immediate, .ptr_stack_offset => {
            const reg = try self.copyToTmpRegister(ty, mcv);
            return self.genSetStack(ty, stack_offset, MCValue{ .register = reg });
        },
        .register => |reg| {
            // MOVE reg, (%sp) + stack_offset
            _ = try self.addInst(.{
                .tag = .MOVE,
                .data = .{ .src_dest = .{
                    .src = .{ .register = reg },
                    .dest = .{ .address_in_register_plus_offset = .{
                        .register = .a7, // sp
                        .offset = stack_offset,
                    }},
                }},
            });
        },
        else => std.debug.panic(
            "TODO: m68k backend: implement genSetStack(???, {}, {})\n",
            .{stack_offset, mcv}
        ),
    }
}

/// Copies a value to a register without tracking the register. The register is not considered
/// allocated. A second call to `copyToTmpRegister` may return the same register.
/// This can have a side effect of spilling instructions to the stack to free up a register.
fn copyToTmpRegister(self: *Self, ty: Type, mcv: MCValue) !Register {
    const reg = try self.register_manager.allocReg(null, gp);
    try self.genSetReg(ty, reg, mcv);
    return reg;
}

fn genSetReg(self: *Self, ty: Type, reg: Register, mcv: MCValue) InnerError!void {
    // const size = ty.abiSize(self.target.*);
    // if (size > 4) {
    //     std.debug.panic(
    //         "m68k backend: genSetReg called with a value larger than one register (32 bits): size was {} bits.\n" ++
    //         "value was '{}'\n",
    //         .{size * 8, mcv},
    //     );
    // }

    switch (mcv) {
        .dead, .unreach => unreachable,
        .none => return,
        .undef => {
            if (self.wantSafety()) {
                // Write the debug undefined value.
                // All m68k registers are 32 bits wide, so it's always 0xAAAAAAAA.
                return self.genSetReg(ty, reg, .{ .immediate = 0xAAAAAAAA });
            }
            // We don't need to do anything if we don't need safety —
            // programs should handle *any* possible register contents!
        },
        .immediate => {
            // OPTIMIZATION: What's the fastest way to set a register to a value?
            // can also add an `if` here to easily set 0xFFFF via an OR instruction

            // Zero the register...
            try self.genZeroRegister(reg);

            // ...and then add the immediate if it's not zero
            if (mcv.immediate != 0) {
                _ = try self.addInst(.{
                    .tag = .ADDI, // Add Immediate
                    .data = .{
                        .immediate_and_register_or_address = .{
                            .immediate = mcv.immediate,
                            .register_or_address = .{ .register = reg },
                        },
                    }
                });
            }
        },
        .register => |src_reg| {
            if (src_reg.id() != reg.id()) {
                // MOVE reg, src_reg
                _ = try self.addInst(.{
                    .tag = .MOVE,
                    .data = .{
                        .src_dest = .{
                            .src = .{ .register = src_reg },
                            .dest = .{ .register = reg },
                        },
                    }
                });
            }
        },
        .ptr_stack_offset => |off| {
            // OPTIMIZATION: can we add the offset to LEA?
            // LEA (%sp), reg
            _ = try self.addInst(.{
                .tag = .LEA,
                .data = .{
                    .register_and_address_mode = .{
                        .address_mode = .{ .address_in_register = .a7 },
                        .register = reg,
                    },
                }
            });

            // ADDI off, reg
            _ = try self.addInst(.{
                .tag = .ADDI,
                .data = .{
                    .immediate_and_register_or_address = .{
                        .immediate = off,
                        .register_or_address = .{ .register = reg },
                    },
                }
            });
        },
        .memory => |addr| {
            // LEA addr, reg
            _ = try self.addInst(.{
                .tag = .LEA,
                .data = .{
                    .register_and_address_mode = .{
                        .address_mode = .{ .address = addr },
                        .register = reg,
                    },
                }
            });
        },

        .stack_offset => |off| {
            // MOVE (%sp) + off, reg
            _ = try self.addInst(.{
                .tag = .MOVE,
                .data = .{
                    .src_dest = .{
                        .src = .{ .address_in_register_plus_offset = .{
                            .register = .a7, // sp
                            .offset = off,
                        }},
                        .dest = .{ .register = reg },
                    },
                }
            });
        },

        // else => std.debug.panic("TODO: m68k backend: implement genSetReg for MCValues like {}", .{ mcv }),
    }
}

fn genZeroRegister(self: *Self, reg: Register) !void {
    // Zero the register by bitwise ANDing it with 0.
    _ = try self.addInst(.{
        .tag = .ANDI, // AND Immediate
        .data = .{
            .immediate_and_register_or_address = .{
                .immediate = 0,
                .register_or_address = .{ .register = reg },
            },
        }
    });
}

/// Jump!
fn jump(self: *Self, inst: Mir.Inst.Index) !void {
    _ = try self.addInst(.{
        .tag = .JMP,
        .data = .{ .inst = inst },
    });
}


fn load(self: *Self, dst_mcv: MCValue, ptr: MCValue, ptr_ty: Type) !void {
    const elem_ty = ptr_ty.elemType();
    switch (ptr) {
        .none => unreachable,
        .undef => unreachable,
        .unreach => unreachable,
        .dead => unreachable,
        .immediate => |imm| try self.setRegOrMem(elem_ty, dst_mcv, .{ .memory = imm }),
        .ptr_stack_offset => |off| try self.setRegOrMem(elem_ty, dst_mcv, .{ .stack_offset = off }),
        .memory, .stack_offset => {
            const reg = try self.register_manager.allocReg(null, gp);
            const reg_lock = self.register_manager.lockRegAssumeUnused(reg);
            defer self.register_manager.unlockReg(reg_lock);

            try self.genSetReg(ptr_ty, reg, ptr);
            try self.load(dst_mcv, .{ .register = reg }, ptr_ty);
        },
        else => std.debug.panic("TODO: m68k backend: implement load for MCValues like {}", .{ ptr }),
    }
}
