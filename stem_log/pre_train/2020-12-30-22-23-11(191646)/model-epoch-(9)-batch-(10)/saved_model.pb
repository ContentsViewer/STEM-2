ох
Юп
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
3
Square
x"T
y"T"
Ttype:
2
	
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
Т
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
Б
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8Н√
l
Adagrad/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdagrad/iter
e
 Adagrad/iter/Read/ReadVariableOpReadVariableOpAdagrad/iter*
_output_shapes
: *
dtype0	
n
Adagrad/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdagrad/decay
g
!Adagrad/decay/Read/ReadVariableOpReadVariableOpAdagrad/decay*
_output_shapes
: *
dtype0
~
Adagrad/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdagrad/learning_rate
w
)Adagrad/learning_rate/Read/ReadVariableOpReadVariableOpAdagrad/learning_rate*
_output_shapes
: *
dtype0
С
LSTM/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*+
shared_nameLSTM/lstm/lstm_cell/kernel
К
.LSTM/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpLSTM/lstm/lstm_cell/kernel*
_output_shapes
:	А*
dtype0
¶
$LSTM/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*5
shared_name&$LSTM/lstm/lstm_cell/recurrent_kernel
Я
8LSTM/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp$LSTM/lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
АА*
dtype0
Й
LSTM/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameLSTM/lstm/lstm_cell/bias
В
,LSTM/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpLSTM/lstm/lstm_cell/bias*
_output_shapes	
:А*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
є
.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*?
shared_name0.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulator
≤
BAdagrad/LSTM/lstm/lstm_cell/kernel/accumulator/Read/ReadVariableOpReadVariableOp.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulator*
_output_shapes
:	А*
dtype0
ќ
8Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*I
shared_name:8Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator
«
LAdagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator* 
_output_shapes
:
АА*
dtype0
±
,Adagrad/LSTM/lstm/lstm_cell/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*=
shared_name.,Adagrad/LSTM/lstm/lstm_cell/bias/accumulator
™
@Adagrad/LSTM/lstm/lstm_cell/bias/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/LSTM/lstm/lstm_cell/bias/accumulator*
_output_shapes	
:А*
dtype0

NoOpNoOp
Ї
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*х
valueлBи Bб
≤
layer-0
layer_with_weights-0
layer-1
layer-2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
 
\

lstm
regularization_losses
trainable_variables
	variables
	keras_api

	keras_api
a
iter
	decay
learning_rateaccumulator:accumulator;accumulator<
 

0
1
2

0
1
2
≠
regularization_losses

layers
non_trainable_variables
layer_metrics
trainable_variables
layer_regularization_losses
metrics
	variables
 
l
cell

state_spec
regularization_losses
trainable_variables
	variables
 	keras_api
 

0
1
2

0
1
2
≠
regularization_losses

!layers
"non_trainable_variables
#layer_metrics
trainable_variables
$layer_regularization_losses
%metrics
	variables
 
KI
VARIABLE_VALUEAdagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEAdagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUELSTM/lstm/lstm_cell/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE$LSTM/lstm/lstm_cell/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUELSTM/lstm/lstm_cell/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
 

&0
~

kernel
recurrent_kernel
bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
 
 

0
1
2

0
1
2
є

+states
regularization_losses

,layers
-non_trainable_variables
.layer_metrics
trainable_variables
/layer_regularization_losses
0metrics
	variables


0
 
 
 
 
4
	1total
	2count
3	variables
4	keras_api
 

0
1
2

0
1
2
≠
'regularization_losses

5layers
6non_trainable_variables
7layer_metrics
(trainable_variables
8layer_regularization_losses
9metrics
)	variables
 

0
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

10
21

3	variables
 
 
 
 
 
ЫШ
VARIABLE_VALUE.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulatorVtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
•Ґ
VARIABLE_VALUE8Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulatorVtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE,Adagrad/LSTM/lstm/lstm_cell/bias/accumulatorVtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
В
serving_default_input_1Placeholder*+
_output_shapes
:€€€€€€€€€d*
dtype0* 
shape:€€€€€€€€€d
Ч
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1LSTM/lstm/lstm_cell/kernel$LSTM/lstm/lstm_cell/recurrent_kernelLSTM/lstm/lstm_cell/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_11856587
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ї
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename Adagrad/iter/Read/ReadVariableOp!Adagrad/decay/Read/ReadVariableOp)Adagrad/learning_rate/Read/ReadVariableOp.LSTM/lstm/lstm_cell/kernel/Read/ReadVariableOp8LSTM/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp,LSTM/lstm/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpBAdagrad/LSTM/lstm/lstm_cell/kernel/accumulator/Read/ReadVariableOpLAdagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator/Read/ReadVariableOp@Adagrad/LSTM/lstm/lstm_cell/bias/accumulator/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_save_11858247
ў
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameAdagrad/iterAdagrad/decayAdagrad/learning_rateLSTM/lstm/lstm_cell/kernel$LSTM/lstm/lstm_cell/recurrent_kernelLSTM/lstm/lstm_cell/biastotalcount.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulator8Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator,Adagrad/LSTM/lstm/lstm_cell/bias/accumulator*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference__traced_restore_11858290ЂГ
є
Ќ
while_cond_11857913
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_11857913___redundant_placeholder06
2while_while_cond_11857913___redundant_placeholder16
2while_while_cond_11857913___redundant_placeholder26
2while_while_cond_11857913___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
Г
Л
'__inference_LSTM_layer_call_fn_11857317

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_LSTM_layer_call_and_return_conditional_losses_118562072
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
–	
±
lstm_while_cond_11857032&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1@
<lstm_while_lstm_while_cond_11857032___redundant_placeholder0@
<lstm_while_lstm_while_cond_11857032___redundant_placeholder1@
<lstm_while_lstm_while_cond_11857032___redundant_placeholder2@
<lstm_while_lstm_while_cond_11857032___redundant_placeholder3
lstm_while_identity
Й
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
ж

Х
LSTM_lstm_while_cond_118566540
,lstm_lstm_while_lstm_lstm_while_loop_counter6
2lstm_lstm_while_lstm_lstm_while_maximum_iterations
lstm_lstm_while_placeholder!
lstm_lstm_while_placeholder_1!
lstm_lstm_while_placeholder_2!
lstm_lstm_while_placeholder_32
.lstm_lstm_while_less_lstm_lstm_strided_slice_1J
Flstm_lstm_while_lstm_lstm_while_cond_11856654___redundant_placeholder0J
Flstm_lstm_while_lstm_lstm_while_cond_11856654___redundant_placeholder1J
Flstm_lstm_while_lstm_lstm_while_cond_11856654___redundant_placeholder2J
Flstm_lstm_while_lstm_lstm_while_cond_11856654___redundant_placeholder3
lstm_lstm_while_identity
Ґ
LSTM/lstm/while/LessLesslstm_lstm_while_placeholder.lstm_lstm_while_less_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LSTM/lstm/while/Less{
LSTM/lstm/while/IdentityIdentityLSTM/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2
LSTM/lstm/while/Identity"=
lstm_lstm_while_identity!LSTM/lstm/while/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
µ@
е
while_body_11857562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceИҐ&while/lstm_cell/BiasAdd/ReadVariableOpҐ%while/lstm_cell/MatMul/ReadVariableOpҐ'while/lstm_cell/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemј
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpќ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul«
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpЈ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul_1ђ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/addњ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpє
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/ConstД
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimГ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
while/lstm_cell/splitЗ
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/TanhЛ
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_1Х
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mulЛ
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_2Ю
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_1Ю
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/add_1Л
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_3Ж
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_4†
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_2Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ў
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityм
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1џ
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2И
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3щ
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_4щ
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
я 
¬
B__inference_LSTM_layer_call_and_return_conditional_losses_11856158
input_1
lstm_11856138
lstm_11856140
lstm_11856142
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐlstm/StatefulPartitionedCallЩ
lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_11856138lstm_11856140lstm_11856142*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_layer_call_and_return_conditional_losses_118559502
lstm/StatefulPartitionedCallЋ
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856138*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulа
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856140* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul°
IdentityIdentity%lstm/StatefulPartitionedCall:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^lstm/StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€d
!
_user_specified_name	input_1
ЃO
•

LSTM_lstm_while_body_118568270
,lstm_lstm_while_lstm_lstm_while_loop_counter6
2lstm_lstm_while_lstm_lstm_while_maximum_iterations
lstm_lstm_while_placeholder!
lstm_lstm_while_placeholder_1!
lstm_lstm_while_placeholder_2!
lstm_lstm_while_placeholder_3/
+lstm_lstm_while_lstm_lstm_strided_slice_1_0k
glstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0>
:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0@
<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0?
;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_lstm_while_identity
lstm_lstm_while_identity_1
lstm_lstm_while_identity_2
lstm_lstm_while_identity_3
lstm_lstm_while_identity_4
lstm_lstm_while_identity_5-
)lstm_lstm_while_lstm_lstm_strided_slice_1i
elstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor<
8lstm_lstm_while_lstm_cell_matmul_readvariableop_resource>
:lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource=
9lstm_lstm_while_lstm_cell_biasadd_readvariableop_resourceИҐ0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpҐ/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpҐ1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp„
ALSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2C
ALSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
3LSTM/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemglstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_lstm_while_placeholderJLSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype025
3LSTM/lstm/while/TensorArrayV2Read/TensorListGetItemё
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype021
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpц
 LSTM/lstm/while/lstm_cell/MatMulMatMul:LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem:item:07LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 LSTM/lstm/while/lstm_cell/MatMulе
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype023
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOpя
"LSTM/lstm/while/lstm_cell/MatMul_1MatMullstm_lstm_while_placeholder_29LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"LSTM/lstm/while/lstm_cell/MatMul_1‘
LSTM/lstm/while/lstm_cell/addAddV2*LSTM/lstm/while/lstm_cell/MatMul:product:0,LSTM/lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/while/lstm_cell/addЁ
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype022
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpб
!LSTM/lstm/while/lstm_cell/BiasAddBiasAdd!LSTM/lstm/while/lstm_cell/add:z:08LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!LSTM/lstm/while/lstm_cell/BiasAddД
LSTM/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
LSTM/lstm/while/lstm_cell/ConstШ
)LSTM/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)LSTM/lstm/while/lstm_cell/split/split_dimЂ
LSTM/lstm/while/lstm_cell/splitSplit2LSTM/lstm/while/lstm_cell/split/split_dim:output:0*LSTM/lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2!
LSTM/lstm/while/lstm_cell/split•
LSTM/lstm/while/lstm_cell/TanhTanh(LSTM/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
LSTM/lstm/while/lstm_cell/Tanh©
 LSTM/lstm/while/lstm_cell/Tanh_1Tanh(LSTM/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2"
 LSTM/lstm/while/lstm_cell/Tanh_1љ
LSTM/lstm/while/lstm_cell/mulMul$LSTM/lstm/while/lstm_cell/Tanh_1:y:0lstm_lstm_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/while/lstm_cell/mul©
 LSTM/lstm/while/lstm_cell/Tanh_2Tanh(LSTM/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2"
 LSTM/lstm/while/lstm_cell/Tanh_2∆
LSTM/lstm/while/lstm_cell/mul_1Mul"LSTM/lstm/while/lstm_cell/Tanh:y:0$LSTM/lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
LSTM/lstm/while/lstm_cell/mul_1∆
LSTM/lstm/while/lstm_cell/add_1AddV2!LSTM/lstm/while/lstm_cell/mul:z:0#LSTM/lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
LSTM/lstm/while/lstm_cell/add_1©
 LSTM/lstm/while/lstm_cell/Tanh_3Tanh(LSTM/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2"
 LSTM/lstm/while/lstm_cell/Tanh_3§
 LSTM/lstm/while/lstm_cell/Tanh_4Tanh#LSTM/lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 LSTM/lstm/while/lstm_cell/Tanh_4»
LSTM/lstm/while/lstm_cell/mul_2Mul$LSTM/lstm/while/lstm_cell/Tanh_3:y:0$LSTM/lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
LSTM/lstm/while/lstm_cell/mul_2П
4LSTM/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_lstm_while_placeholder_1lstm_lstm_while_placeholder#LSTM/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype026
4LSTM/lstm/while/TensorArrayV2Write/TensorListSetItemp
LSTM/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/while/add/yС
LSTM/lstm/while/addAddV2lstm_lstm_while_placeholderLSTM/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/while/addt
LSTM/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/while/add_1/y®
LSTM/lstm/while/add_1AddV2,lstm_lstm_while_lstm_lstm_while_loop_counter LSTM/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/while/add_1Х
LSTM/lstm/while/IdentityIdentityLSTM/lstm/while/add_1:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity≤
LSTM/lstm/while/Identity_1Identity2lstm_lstm_while_lstm_lstm_while_maximum_iterations1^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_1Ч
LSTM/lstm/while/Identity_2IdentityLSTM/lstm/while/add:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_2ƒ
LSTM/lstm/while/Identity_3IdentityDLSTM/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_3µ
LSTM/lstm/while/Identity_4Identity#LSTM/lstm/while/lstm_cell/mul_2:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/while/Identity_4µ
LSTM/lstm/while/Identity_5Identity#LSTM/lstm/while/lstm_cell/add_1:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/while/Identity_5"=
lstm_lstm_while_identity!LSTM/lstm/while/Identity:output:0"A
lstm_lstm_while_identity_1#LSTM/lstm/while/Identity_1:output:0"A
lstm_lstm_while_identity_2#LSTM/lstm/while/Identity_2:output:0"A
lstm_lstm_while_identity_3#LSTM/lstm/while/Identity_3:output:0"A
lstm_lstm_while_identity_4#LSTM/lstm/while/Identity_4:output:0"A
lstm_lstm_while_identity_5#LSTM/lstm/while/Identity_5:output:0"x
9lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"z
:lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"v
8lstm_lstm_while_lstm_cell_matmul_readvariableop_resource:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0"X
)lstm_lstm_while_lstm_lstm_strided_slice_1+lstm_lstm_while_lstm_lstm_strided_slice_1_0"–
elstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensorglstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2d
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp2b
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2f
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
≈r
з
B__inference_lstm_layer_call_and_return_conditional_losses_11857659

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ lstm_cell/BiasAdd/ReadVariableOpҐlstm_cell/MatMul/ReadVariableOpҐ!lstm_cell/MatMul_1/ReadVariableOpҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d€€€€€€€€€2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ђ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02!
lstm_cell/MatMul/ReadVariableOp§
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul≥
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp†
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul_1Ф
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/addЂ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp°
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimл
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_1А
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_2Ж
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_1Ж
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_4И
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterн
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_11857562*
condR
while_cond_11857561*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d€€€€€€€€€А*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€dА2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeж
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulэ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulж
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
Е
М
(__inference_model_layer_call_fn_11856953

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_118565492
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
С,
√
C__inference_model_layer_call_and_return_conditional_losses_11856445
input_1
lstm_11856418
lstm_11856420
lstm_11856422
identityИҐLSTM/StatefulPartitionedCallҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЩ
LSTM/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_11856418lstm_11856420lstm_11856422*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_LSTM_layer_call_and_return_conditional_losses_118563952
LSTM/StatefulPartitionedCallЄ
(tf.math.l2_normalize/l2_normalize/SquareSquare%LSTM/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(tf.math.l2_normalize/l2_normalize/Squareі
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesИ
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/SumЯ
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *€жџ.2-
+tf.math.l2_normalize/l2_normalize/Maximum/yщ
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2+
)tf.math.l2_normalize/l2_normalize/MaximumЉ
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2)
'tf.math.l2_normalize/l2_normalize/Rsqrt‘
!tf.math.l2_normalize/l2_normalizeMul%LSTM/StatefulPartitionedCall:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!tf.math.l2_normalize/l2_normalizeЋ
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856418*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulа
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856420* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul°
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0^LSTM/StatefulPartitionedCall=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2<
LSTM/StatefulPartitionedCallLSTM/StatefulPartitionedCall2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:€€€€€€€€€d
!
_user_specified_name	input_1
¶|
К
B__inference_LSTM_layer_call_and_return_conditional_losses_11856395

inputs1
-lstm_lstm_cell_matmul_readvariableop_resource3
/lstm_lstm_cell_matmul_1_readvariableop_resource2
.lstm_lstm_cell_biasadd_readvariableop_resource
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ%lstm/lstm_cell/BiasAdd/ReadVariableOpҐ$lstm/lstm_cell/MatMul/ReadVariableOpҐ&lstm/lstm_cell/MatMul_1/ReadVariableOpҐ
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stackВ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1В
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2А
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_sliceg
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros/mul/yА
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessm
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros/packed/1Ч
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/ConstК

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

lstm/zerosk
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros_1/mul/yЖ
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm/zeros_1/Less/yГ
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessq
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros_1/packed/1Э
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/ConstТ
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/permЙ
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:d€€€€€€€€€2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1В
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stackЖ
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1Ж
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2М
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1П
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2"
 lstm/TensorArrayV2/element_shape∆
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2…
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensorВ
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stackЖ
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1Ж
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2Ъ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm/strided_slice_2ї
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOpЄ
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/MatMul¬
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpі
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/MatMul_1®
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/addЇ
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOpµ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/BiasAddn
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/ConstВ
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim€
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm/lstm_cell/splitД
lstm/lstm_cell/TanhTanhlstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/TanhИ
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_1Ф
lstm/lstm_cell/mulMullstm/lstm_cell/Tanh_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/mulИ
lstm/lstm_cell/Tanh_2Tanhlstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_2Ъ
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Tanh:y:0lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/mul_1Ъ
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/add_1И
lstm/lstm_cell/Tanh_3Tanhlstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_3Г
lstm/lstm_cell/Tanh_4Tanhlstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_4Ь
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Tanh_3:y:0lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/mul_2Щ
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2$
"lstm/TensorArrayV2_1/element_shapeћ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/timeЙ
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterЄ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_while_body_11856298*$
condR
lstm_while_cond_11856297*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2

lstm/whileњ
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeэ
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d€€€€€€€€€А*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStackЛ
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm/strided_slice_3/stackЖ
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1Ж
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2є
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
lstm/strided_slice_3Г
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permЇ
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€dА2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeл
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulВ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul€
IdentityIdentitylstm/strided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
µ@
е
while_body_11855853
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceИҐ&while/lstm_cell/BiasAdd/ReadVariableOpҐ%while/lstm_cell/MatMul/ReadVariableOpҐ'while/lstm_cell/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemј
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpќ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul«
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpЈ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul_1ђ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/addњ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpє
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/ConstД
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimГ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
while/lstm_cell/splitЗ
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/TanhЛ
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_1Х
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mulЛ
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_2Ю
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_1Ю
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/add_1Л
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_3Ж
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_4†
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_2Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ў
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityм
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1џ
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2И
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3щ
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_4щ
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
О,
¬
C__inference_model_layer_call_and_return_conditional_losses_11856508

inputs
lstm_11856481
lstm_11856483
lstm_11856485
identityИҐLSTM/StatefulPartitionedCallҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpШ
LSTM/StatefulPartitionedCallStatefulPartitionedCallinputslstm_11856481lstm_11856483lstm_11856485*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_LSTM_layer_call_and_return_conditional_losses_118563952
LSTM/StatefulPartitionedCallЄ
(tf.math.l2_normalize/l2_normalize/SquareSquare%LSTM/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(tf.math.l2_normalize/l2_normalize/Squareі
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesИ
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/SumЯ
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *€жџ.2-
+tf.math.l2_normalize/l2_normalize/Maximum/yщ
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2+
)tf.math.l2_normalize/l2_normalize/MaximumЉ
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2)
'tf.math.l2_normalize/l2_normalize/Rsqrt‘
!tf.math.l2_normalize/l2_normalizeMul%LSTM/StatefulPartitionedCall:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!tf.math.l2_normalize/l2_normalizeЋ
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856481*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulа
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856483* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul°
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0^LSTM/StatefulPartitionedCall=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2<
LSTM/StatefulPartitionedCallLSTM/StatefulPartitionedCall2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
»7
д
G__inference_lstm_cell_layer_call_and_return_conditional_losses_11858135

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim√
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
splitW
TanhTanhsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh[
Tanh_1Tanhsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_1Z
mulMul
Tanh_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€А2
mul[
Tanh_2Tanhsplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_2^
mul_1MulTanh:y:0
Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
add_1[
Tanh_3Tanhsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_3V
Tanh_4Tanh	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_4`
mul_2Mul
Tanh_3:y:0
Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mul_2№
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulу
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul±
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityµ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1µ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
states/0:RN
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
states/1
Ы
Н
'__inference_lstm_layer_call_fn_11858022
inputs_0
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_layer_call_and_return_conditional_losses_118556292
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ї%
щ
!__inference__traced_save_11858247
file_prefix+
'savev2_adagrad_iter_read_readvariableop	,
(savev2_adagrad_decay_read_readvariableop4
0savev2_adagrad_learning_rate_read_readvariableop9
5savev2_lstm_lstm_lstm_cell_kernel_read_readvariableopC
?savev2_lstm_lstm_lstm_cell_recurrent_kernel_read_readvariableop7
3savev2_lstm_lstm_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopM
Isavev2_adagrad_lstm_lstm_lstm_cell_kernel_accumulator_read_readvariableopW
Ssavev2_adagrad_lstm_lstm_lstm_cell_recurrent_kernel_accumulator_read_readvariableopK
Gsavev2_adagrad_lstm_lstm_lstm_cell_bias_accumulator_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameґ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*»
valueЊBїB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names†
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices§
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_adagrad_iter_read_readvariableop(savev2_adagrad_decay_read_readvariableop0savev2_adagrad_learning_rate_read_readvariableop5savev2_lstm_lstm_lstm_cell_kernel_read_readvariableop?savev2_lstm_lstm_lstm_cell_recurrent_kernel_read_readvariableop3savev2_lstm_lstm_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopIsavev2_adagrad_lstm_lstm_lstm_cell_kernel_accumulator_read_readvariableopSsavev2_adagrad_lstm_lstm_lstm_cell_recurrent_kernel_accumulator_read_readvariableopGsavev2_adagrad_lstm_lstm_lstm_cell_bias_accumulator_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*]
_input_shapesL
J: : : : :	А:
АА:А: : :	А:
АА:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:

_output_shapes
: :

_output_shapes
: :%	!

_output_shapes
:	А:&
"
 
_output_shapes
:
АА:!

_output_shapes	
:А:

_output_shapes
: 
≈r
з
B__inference_lstm_layer_call_and_return_conditional_losses_11856115

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ lstm_cell/BiasAdd/ReadVariableOpҐlstm_cell/MatMul/ReadVariableOpҐ!lstm_cell/MatMul_1/ReadVariableOpҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d€€€€€€€€€2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ђ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02!
lstm_cell/MatMul/ReadVariableOp§
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul≥
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp†
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul_1Ф
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/addЂ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp°
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimл
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_1А
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_2Ж
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_1Ж
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_4И
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterн
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_11856018*
condR
while_cond_11856017*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d€€€€€€€€€А*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€dА2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeж
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulэ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulж
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
ј7
в
G__inference_lstm_cell_layer_call_and_return_conditional_losses_11855185

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim√
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
splitW
TanhTanhsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh[
Tanh_1Tanhsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_1Z
mulMul
Tanh_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€А2
mul[
Tanh_2Tanhsplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_2^
mul_1MulTanh:y:0
Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
add_1[
Tanh_3Tanhsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_3V
Tanh_4Tanh	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_4`
mul_2Mul
Tanh_3:y:0
Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mul_2№
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulу
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul±
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityµ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1µ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_namestates:PL
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_namestates
И
Н
(__inference_model_layer_call_fn_11856517
input_1
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_118565082
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€d
!
_user_specified_name	input_1
Е
М
(__inference_model_layer_call_fn_11856942

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_118565082
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
сG
Е	
lstm_while_body_11857198&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_09
5lstm_while_lstm_cell_matmul_readvariableop_resource_0;
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor7
3lstm_while_lstm_cell_matmul_readvariableop_resource9
5lstm_while_lstm_cell_matmul_1_readvariableop_resource8
4lstm_while_lstm_cell_biasadd_readvariableop_resourceИҐ+lstm/while/lstm_cell/BiasAdd/ReadVariableOpҐ*lstm/while/lstm_cell/MatMul/ReadVariableOpҐ,lstm/while/lstm_cell/MatMul_1/ReadVariableOpЌ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeс
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemѕ
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpв
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/MatMul÷
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpЋ
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/MatMul_1ј
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/addќ
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpЌ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/BiasAddz
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/ConstО
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimЧ
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm/while/lstm_cell/splitЦ
lstm/while/lstm_cell/TanhTanh#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/TanhЪ
lstm/while/lstm_cell/Tanh_1Tanh#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_1©
lstm/while/lstm_cell/mulMullstm/while/lstm_cell/Tanh_1:y:0lstm_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/mulЪ
lstm/while/lstm_cell/Tanh_2Tanh#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_2≤
lstm/while/lstm_cell/mul_1Mullstm/while/lstm_cell/Tanh:y:0lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/mul_1≤
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/add_1Ъ
lstm/while/lstm_cell/Tanh_3Tanh#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_3Х
lstm/while/lstm_cell/Tanh_4Tanhlstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_4і
lstm/while/lstm_cell/mul_2Mullstm/while/lstm_cell/Tanh_3:y:0lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/mul_2ц
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/yП
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1ч
lstm/while/IdentityIdentitylstm/while/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/IdentityП
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1щ
lstm/while/Identity_2Identitylstm/while/add:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2¶
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3Ч
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/Identity_4Ч
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"Љ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
шГ
±
#__inference__wrapped_model_11855100
input_1<
8model_lstm_lstm_lstm_cell_matmul_readvariableop_resource>
:model_lstm_lstm_lstm_cell_matmul_1_readvariableop_resource=
9model_lstm_lstm_lstm_cell_biasadd_readvariableop_resource
identityИҐ0model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOpҐ/model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOpҐ1model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOpҐmodel/LSTM/lstm/whilee
model/LSTM/lstm/ShapeShapeinput_1*
T0*
_output_shapes
:2
model/LSTM/lstm/ShapeФ
#model/LSTM/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model/LSTM/lstm/strided_slice/stackШ
%model/LSTM/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/LSTM/lstm/strided_slice/stack_1Ш
%model/LSTM/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/LSTM/lstm/strided_slice/stack_2¬
model/LSTM/lstm/strided_sliceStridedSlicemodel/LSTM/lstm/Shape:output:0,model/LSTM/lstm/strided_slice/stack:output:0.model/LSTM/lstm/strided_slice/stack_1:output:0.model/LSTM/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/LSTM/lstm/strided_slice}
model/LSTM/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
model/LSTM/lstm/zeros/mul/yђ
model/LSTM/lstm/zeros/mulMul&model/LSTM/lstm/strided_slice:output:0$model/LSTM/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/zeros/mul
model/LSTM/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
model/LSTM/lstm/zeros/Less/yІ
model/LSTM/lstm/zeros/LessLessmodel/LSTM/lstm/zeros/mul:z:0%model/LSTM/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/zeros/LessГ
model/LSTM/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2 
model/LSTM/lstm/zeros/packed/1√
model/LSTM/lstm/zeros/packedPack&model/LSTM/lstm/strided_slice:output:0'model/LSTM/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/LSTM/lstm/zeros/packed
model/LSTM/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/LSTM/lstm/zeros/Constґ
model/LSTM/lstm/zerosFill%model/LSTM/lstm/zeros/packed:output:0$model/LSTM/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/LSTM/lstm/zerosБ
model/LSTM/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
model/LSTM/lstm/zeros_1/mul/y≤
model/LSTM/lstm/zeros_1/mulMul&model/LSTM/lstm/strided_slice:output:0&model/LSTM/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/zeros_1/mulГ
model/LSTM/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
model/LSTM/lstm/zeros_1/Less/yѓ
model/LSTM/lstm/zeros_1/LessLessmodel/LSTM/lstm/zeros_1/mul:z:0'model/LSTM/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/zeros_1/LessЗ
 model/LSTM/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2"
 model/LSTM/lstm/zeros_1/packed/1…
model/LSTM/lstm/zeros_1/packedPack&model/LSTM/lstm/strided_slice:output:0)model/LSTM/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model/LSTM/lstm/zeros_1/packedГ
model/LSTM/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/LSTM/lstm/zeros_1/ConstЊ
model/LSTM/lstm/zeros_1Fill'model/LSTM/lstm/zeros_1/packed:output:0&model/LSTM/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/LSTM/lstm/zeros_1Х
model/LSTM/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model/LSTM/lstm/transpose/permЂ
model/LSTM/lstm/transpose	Transposeinput_1'model/LSTM/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:d€€€€€€€€€2
model/LSTM/lstm/transpose
model/LSTM/lstm/Shape_1Shapemodel/LSTM/lstm/transpose:y:0*
T0*
_output_shapes
:2
model/LSTM/lstm/Shape_1Ш
%model/LSTM/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model/LSTM/lstm/strided_slice_1/stackЬ
'model/LSTM/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/LSTM/lstm/strided_slice_1/stack_1Ь
'model/LSTM/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/LSTM/lstm/strided_slice_1/stack_2ќ
model/LSTM/lstm/strided_slice_1StridedSlice model/LSTM/lstm/Shape_1:output:0.model/LSTM/lstm/strided_slice_1/stack:output:00model/LSTM/lstm/strided_slice_1/stack_1:output:00model/LSTM/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model/LSTM/lstm/strided_slice_1•
+model/LSTM/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2-
+model/LSTM/lstm/TensorArrayV2/element_shapeт
model/LSTM/lstm/TensorArrayV2TensorListReserve4model/LSTM/lstm/TensorArrayV2/element_shape:output:0(model/LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/LSTM/lstm/TensorArrayV2я
Emodel/LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2G
Emodel/LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeЄ
7model/LSTM/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/LSTM/lstm/transpose:y:0Nmodel/LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7model/LSTM/lstm/TensorArrayUnstack/TensorListFromTensorШ
%model/LSTM/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model/LSTM/lstm/strided_slice_2/stackЬ
'model/LSTM/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/LSTM/lstm/strided_slice_2/stack_1Ь
'model/LSTM/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/LSTM/lstm/strided_slice_2/stack_2№
model/LSTM/lstm/strided_slice_2StridedSlicemodel/LSTM/lstm/transpose:y:0.model/LSTM/lstm/strided_slice_2/stack:output:00model/LSTM/lstm/strided_slice_2/stack_1:output:00model/LSTM/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2!
model/LSTM/lstm/strided_slice_2№
/model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp8model_lstm_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype021
/model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOpд
 model/LSTM/lstm/lstm_cell/MatMulMatMul(model/LSTM/lstm/strided_slice_2:output:07model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 model/LSTM/lstm/lstm_cell/MatMulг
1model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:model_lstm_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOpа
"model/LSTM/lstm/lstm_cell/MatMul_1MatMulmodel/LSTM/lstm/zeros:output:09model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"model/LSTM/lstm/lstm_cell/MatMul_1‘
model/LSTM/lstm/lstm_cell/addAddV2*model/LSTM/lstm/lstm_cell/MatMul:product:0,model/LSTM/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/LSTM/lstm/lstm_cell/addџ
0model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9model_lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOpб
!model/LSTM/lstm/lstm_cell/BiasAddBiasAdd!model/LSTM/lstm/lstm_cell/add:z:08model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!model/LSTM/lstm/lstm_cell/BiasAddД
model/LSTM/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
model/LSTM/lstm/lstm_cell/ConstШ
)model/LSTM/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model/LSTM/lstm/lstm_cell/split/split_dimЂ
model/LSTM/lstm/lstm_cell/splitSplit2model/LSTM/lstm/lstm_cell/split/split_dim:output:0*model/LSTM/lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2!
model/LSTM/lstm/lstm_cell/split•
model/LSTM/lstm/lstm_cell/TanhTanh(model/LSTM/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
model/LSTM/lstm/lstm_cell/Tanh©
 model/LSTM/lstm/lstm_cell/Tanh_1Tanh(model/LSTM/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2"
 model/LSTM/lstm/lstm_cell/Tanh_1ј
model/LSTM/lstm/lstm_cell/mulMul$model/LSTM/lstm/lstm_cell/Tanh_1:y:0 model/LSTM/lstm/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/LSTM/lstm/lstm_cell/mul©
 model/LSTM/lstm/lstm_cell/Tanh_2Tanh(model/LSTM/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2"
 model/LSTM/lstm/lstm_cell/Tanh_2∆
model/LSTM/lstm/lstm_cell/mul_1Mul"model/LSTM/lstm/lstm_cell/Tanh:y:0$model/LSTM/lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
model/LSTM/lstm/lstm_cell/mul_1∆
model/LSTM/lstm/lstm_cell/add_1AddV2!model/LSTM/lstm/lstm_cell/mul:z:0#model/LSTM/lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
model/LSTM/lstm/lstm_cell/add_1©
 model/LSTM/lstm/lstm_cell/Tanh_3Tanh(model/LSTM/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2"
 model/LSTM/lstm/lstm_cell/Tanh_3§
 model/LSTM/lstm/lstm_cell/Tanh_4Tanh#model/LSTM/lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 model/LSTM/lstm/lstm_cell/Tanh_4»
model/LSTM/lstm/lstm_cell/mul_2Mul$model/LSTM/lstm/lstm_cell/Tanh_3:y:0$model/LSTM/lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
model/LSTM/lstm/lstm_cell/mul_2ѓ
-model/LSTM/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2/
-model/LSTM/lstm/TensorArrayV2_1/element_shapeш
model/LSTM/lstm/TensorArrayV2_1TensorListReserve6model/LSTM/lstm/TensorArrayV2_1/element_shape:output:0(model/LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
model/LSTM/lstm/TensorArrayV2_1n
model/LSTM/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model/LSTM/lstm/timeЯ
(model/LSTM/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2*
(model/LSTM/lstm/while/maximum_iterationsК
"model/LSTM/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/LSTM/lstm/while/loop_counterЁ
model/LSTM/lstm/whileWhile+model/LSTM/lstm/while/loop_counter:output:01model/LSTM/lstm/while/maximum_iterations:output:0model/LSTM/lstm/time:output:0(model/LSTM/lstm/TensorArrayV2_1:handle:0model/LSTM/lstm/zeros:output:0 model/LSTM/lstm/zeros_1:output:0(model/LSTM/lstm/strided_slice_1:output:0Gmodel/LSTM/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:08model_lstm_lstm_lstm_cell_matmul_readvariableop_resource:model_lstm_lstm_lstm_cell_matmul_1_readvariableop_resource9model_lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#model_LSTM_lstm_while_body_11855008*/
cond'R%
#model_LSTM_lstm_while_cond_11855007*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
model/LSTM/lstm/while’
@model/LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2B
@model/LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shape©
2model/LSTM/lstm/TensorArrayV2Stack/TensorListStackTensorListStackmodel/LSTM/lstm/while:output:3Imodel/LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d€€€€€€€€€А*
element_dtype024
2model/LSTM/lstm/TensorArrayV2Stack/TensorListStack°
%model/LSTM/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2'
%model/LSTM/lstm/strided_slice_3/stackЬ
'model/LSTM/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'model/LSTM/lstm/strided_slice_3/stack_1Ь
'model/LSTM/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/LSTM/lstm/strided_slice_3/stack_2ы
model/LSTM/lstm/strided_slice_3StridedSlice;model/LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.model/LSTM/lstm/strided_slice_3/stack:output:00model/LSTM/lstm/strided_slice_3/stack_1:output:00model/LSTM/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2!
model/LSTM/lstm/strided_slice_3Щ
 model/LSTM/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model/LSTM/lstm/transpose_1/permж
model/LSTM/lstm/transpose_1	Transpose;model/LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)model/LSTM/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€dА2
model/LSTM/lstm/transpose_1Ж
model/LSTM/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model/LSTM/lstm/runtime«
.model/tf.math.l2_normalize/l2_normalize/SquareSquare(model/LSTM/lstm/strided_slice_3:output:0*
T0*(
_output_shapes
:€€€€€€€€€А20
.model/tf.math.l2_normalize/l2_normalize/Squareј
=model/tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=model/tf.math.l2_normalize/l2_normalize/Sum/reduction_indices†
+model/tf.math.l2_normalize/l2_normalize/SumSum2model/tf.math.l2_normalize/l2_normalize/Square:y:0Fmodel/tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(2-
+model/tf.math.l2_normalize/l2_normalize/SumЂ
1model/tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *€жџ.23
1model/tf.math.l2_normalize/l2_normalize/Maximum/yС
/model/tf.math.l2_normalize/l2_normalize/MaximumMaximum4model/tf.math.l2_normalize/l2_normalize/Sum:output:0:model/tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€21
/model/tf.math.l2_normalize/l2_normalize/Maximumќ
-model/tf.math.l2_normalize/l2_normalize/RsqrtRsqrt3model/tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2/
-model/tf.math.l2_normalize/l2_normalize/Rsqrtй
'model/tf.math.l2_normalize/l2_normalizeMul(model/LSTM/lstm/strided_slice_3:output:01model/tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2)
'model/tf.math.l2_normalize/l2_normalize±
IdentityIdentity+model/tf.math.l2_normalize/l2_normalize:z:01^model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp0^model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOp2^model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp^model/LSTM/lstm/while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2d
0model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp0model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp2b
/model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOp/model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOp2f
1model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp1model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp2.
model/LSTM/lstm/whilemodel/LSTM/lstm/while:T P
+
_output_shapes
:€€€€€€€€€d
!
_user_specified_name	input_1
ЃO
•

LSTM_lstm_while_body_118566550
,lstm_lstm_while_lstm_lstm_while_loop_counter6
2lstm_lstm_while_lstm_lstm_while_maximum_iterations
lstm_lstm_while_placeholder!
lstm_lstm_while_placeholder_1!
lstm_lstm_while_placeholder_2!
lstm_lstm_while_placeholder_3/
+lstm_lstm_while_lstm_lstm_strided_slice_1_0k
glstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0>
:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0@
<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0?
;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_lstm_while_identity
lstm_lstm_while_identity_1
lstm_lstm_while_identity_2
lstm_lstm_while_identity_3
lstm_lstm_while_identity_4
lstm_lstm_while_identity_5-
)lstm_lstm_while_lstm_lstm_strided_slice_1i
elstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor<
8lstm_lstm_while_lstm_cell_matmul_readvariableop_resource>
:lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource=
9lstm_lstm_while_lstm_cell_biasadd_readvariableop_resourceИҐ0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpҐ/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpҐ1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp„
ALSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2C
ALSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
3LSTM/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemglstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_lstm_while_placeholderJLSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype025
3LSTM/lstm/while/TensorArrayV2Read/TensorListGetItemё
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype021
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpц
 LSTM/lstm/while/lstm_cell/MatMulMatMul:LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem:item:07LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 LSTM/lstm/while/lstm_cell/MatMulе
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype023
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOpя
"LSTM/lstm/while/lstm_cell/MatMul_1MatMullstm_lstm_while_placeholder_29LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"LSTM/lstm/while/lstm_cell/MatMul_1‘
LSTM/lstm/while/lstm_cell/addAddV2*LSTM/lstm/while/lstm_cell/MatMul:product:0,LSTM/lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/while/lstm_cell/addЁ
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype022
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpб
!LSTM/lstm/while/lstm_cell/BiasAddBiasAdd!LSTM/lstm/while/lstm_cell/add:z:08LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!LSTM/lstm/while/lstm_cell/BiasAddД
LSTM/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
LSTM/lstm/while/lstm_cell/ConstШ
)LSTM/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)LSTM/lstm/while/lstm_cell/split/split_dimЂ
LSTM/lstm/while/lstm_cell/splitSplit2LSTM/lstm/while/lstm_cell/split/split_dim:output:0*LSTM/lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2!
LSTM/lstm/while/lstm_cell/split•
LSTM/lstm/while/lstm_cell/TanhTanh(LSTM/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
LSTM/lstm/while/lstm_cell/Tanh©
 LSTM/lstm/while/lstm_cell/Tanh_1Tanh(LSTM/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2"
 LSTM/lstm/while/lstm_cell/Tanh_1љ
LSTM/lstm/while/lstm_cell/mulMul$LSTM/lstm/while/lstm_cell/Tanh_1:y:0lstm_lstm_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/while/lstm_cell/mul©
 LSTM/lstm/while/lstm_cell/Tanh_2Tanh(LSTM/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2"
 LSTM/lstm/while/lstm_cell/Tanh_2∆
LSTM/lstm/while/lstm_cell/mul_1Mul"LSTM/lstm/while/lstm_cell/Tanh:y:0$LSTM/lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
LSTM/lstm/while/lstm_cell/mul_1∆
LSTM/lstm/while/lstm_cell/add_1AddV2!LSTM/lstm/while/lstm_cell/mul:z:0#LSTM/lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
LSTM/lstm/while/lstm_cell/add_1©
 LSTM/lstm/while/lstm_cell/Tanh_3Tanh(LSTM/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2"
 LSTM/lstm/while/lstm_cell/Tanh_3§
 LSTM/lstm/while/lstm_cell/Tanh_4Tanh#LSTM/lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 LSTM/lstm/while/lstm_cell/Tanh_4»
LSTM/lstm/while/lstm_cell/mul_2Mul$LSTM/lstm/while/lstm_cell/Tanh_3:y:0$LSTM/lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
LSTM/lstm/while/lstm_cell/mul_2П
4LSTM/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_lstm_while_placeholder_1lstm_lstm_while_placeholder#LSTM/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype026
4LSTM/lstm/while/TensorArrayV2Write/TensorListSetItemp
LSTM/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/while/add/yС
LSTM/lstm/while/addAddV2lstm_lstm_while_placeholderLSTM/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/while/addt
LSTM/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/while/add_1/y®
LSTM/lstm/while/add_1AddV2,lstm_lstm_while_lstm_lstm_while_loop_counter LSTM/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/while/add_1Х
LSTM/lstm/while/IdentityIdentityLSTM/lstm/while/add_1:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity≤
LSTM/lstm/while/Identity_1Identity2lstm_lstm_while_lstm_lstm_while_maximum_iterations1^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_1Ч
LSTM/lstm/while/Identity_2IdentityLSTM/lstm/while/add:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_2ƒ
LSTM/lstm/while/Identity_3IdentityDLSTM/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_3µ
LSTM/lstm/while/Identity_4Identity#LSTM/lstm/while/lstm_cell/mul_2:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/while/Identity_4µ
LSTM/lstm/while/Identity_5Identity#LSTM/lstm/while/lstm_cell/add_1:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/while/Identity_5"=
lstm_lstm_while_identity!LSTM/lstm/while/Identity:output:0"A
lstm_lstm_while_identity_1#LSTM/lstm/while/Identity_1:output:0"A
lstm_lstm_while_identity_2#LSTM/lstm/while/Identity_2:output:0"A
lstm_lstm_while_identity_3#LSTM/lstm/while/Identity_3:output:0"A
lstm_lstm_while_identity_4#LSTM/lstm/while/Identity_4:output:0"A
lstm_lstm_while_identity_5#LSTM/lstm/while/Identity_5:output:0"x
9lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"z
:lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"v
8lstm_lstm_while_lstm_cell_matmul_readvariableop_resource:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0"X
)lstm_lstm_while_lstm_lstm_strided_slice_1+lstm_lstm_while_lstm_lstm_strided_slice_1_0"–
elstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensorglstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2d
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp2b
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2f
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
¶|
К
B__inference_LSTM_layer_call_and_return_conditional_losses_11857295

inputs1
-lstm_lstm_cell_matmul_readvariableop_resource3
/lstm_lstm_cell_matmul_1_readvariableop_resource2
.lstm_lstm_cell_biasadd_readvariableop_resource
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ%lstm/lstm_cell/BiasAdd/ReadVariableOpҐ$lstm/lstm_cell/MatMul/ReadVariableOpҐ&lstm/lstm_cell/MatMul_1/ReadVariableOpҐ
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stackВ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1В
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2А
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_sliceg
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros/mul/yА
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessm
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros/packed/1Ч
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/ConstК

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

lstm/zerosk
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros_1/mul/yЖ
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm/zeros_1/Less/yГ
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessq
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros_1/packed/1Э
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/ConstТ
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/permЙ
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:d€€€€€€€€€2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1В
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stackЖ
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1Ж
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2М
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1П
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2"
 lstm/TensorArrayV2/element_shape∆
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2…
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensorВ
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stackЖ
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1Ж
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2Ъ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm/strided_slice_2ї
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOpЄ
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/MatMul¬
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpі
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/MatMul_1®
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/addЇ
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOpµ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/BiasAddn
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/ConstВ
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim€
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm/lstm_cell/splitД
lstm/lstm_cell/TanhTanhlstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/TanhИ
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_1Ф
lstm/lstm_cell/mulMullstm/lstm_cell/Tanh_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/mulИ
lstm/lstm_cell/Tanh_2Tanhlstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_2Ъ
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Tanh:y:0lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/mul_1Ъ
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/add_1И
lstm/lstm_cell/Tanh_3Tanhlstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_3Г
lstm/lstm_cell/Tanh_4Tanhlstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_4Ь
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Tanh_3:y:0lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/mul_2Щ
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2$
"lstm/TensorArrayV2_1/element_shapeћ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/timeЙ
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterЄ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_while_body_11857198*$
condR
lstm_while_cond_11857197*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2

lstm/whileњ
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeэ
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d€€€€€€€€€А*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStackЛ
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm/strided_slice_3/stackЖ
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1Ж
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2є
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
lstm/strided_slice_3Г
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permЇ
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€dА2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeл
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulВ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul€
IdentityIdentitylstm/strided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
№]
Ё
B__inference_lstm_layer_call_and_return_conditional_losses_11855773

inputs
lstm_cell_11855679
lstm_cell_11855681
lstm_cell_11855683
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ!lstm_cell/StatefulPartitionedCallҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2Ц
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_11855679lstm_cell_11855681lstm_cell_11855683*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_118552302#
!lstm_cell/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter®
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_11855679lstm_cell_11855681lstm_cell_11855683*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_11855692*
condR
while_cond_11855691*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   22
0TensorArrayV2Stack/TensorListStack/element_shapeт
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permѓ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime–
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_11855679*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulе
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_11855681* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul°
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
–	
±
lstm_while_cond_11857197&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1@
<lstm_while_lstm_while_cond_11857197___redundant_placeholder0@
<lstm_while_lstm_while_cond_11857197___redundant_placeholder1@
<lstm_while_lstm_while_cond_11857197___redundant_placeholder2@
<lstm_while_lstm_while_cond_11857197___redundant_placeholder3
lstm_while_identity
Й
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
°С
Ѓ
C__inference_model_layer_call_and_return_conditional_losses_11856931

inputs6
2lstm_lstm_lstm_cell_matmul_readvariableop_resource8
4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource7
3lstm_lstm_lstm_cell_biasadd_readvariableop_resource
identityИҐ*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOpҐ)LSTM/lstm/lstm_cell/MatMul/ReadVariableOpҐ+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOpҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐLSTM/lstm/whileX
LSTM/lstm/ShapeShapeinputs*
T0*
_output_shapes
:2
LSTM/lstm/ShapeИ
LSTM/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
LSTM/lstm/strided_slice/stackМ
LSTM/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
LSTM/lstm/strided_slice/stack_1М
LSTM/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
LSTM/lstm/strided_slice/stack_2Ю
LSTM/lstm/strided_sliceStridedSliceLSTM/lstm/Shape:output:0&LSTM/lstm/strided_slice/stack:output:0(LSTM/lstm/strided_slice/stack_1:output:0(LSTM/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM/lstm/strided_sliceq
LSTM/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
LSTM/lstm/zeros/mul/yФ
LSTM/lstm/zeros/mulMul LSTM/lstm/strided_slice:output:0LSTM/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros/muls
LSTM/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
LSTM/lstm/zeros/Less/yП
LSTM/lstm/zeros/LessLessLSTM/lstm/zeros/mul:z:0LSTM/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros/Lessw
LSTM/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
LSTM/lstm/zeros/packed/1Ђ
LSTM/lstm/zeros/packedPack LSTM/lstm/strided_slice:output:0!LSTM/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM/lstm/zeros/packeds
LSTM/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/zeros/ConstЮ
LSTM/lstm/zerosFillLSTM/lstm/zeros/packed:output:0LSTM/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/zerosu
LSTM/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
LSTM/lstm/zeros_1/mul/yЪ
LSTM/lstm/zeros_1/mulMul LSTM/lstm/strided_slice:output:0 LSTM/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros_1/mulw
LSTM/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
LSTM/lstm/zeros_1/Less/yЧ
LSTM/lstm/zeros_1/LessLessLSTM/lstm/zeros_1/mul:z:0!LSTM/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros_1/Less{
LSTM/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
LSTM/lstm/zeros_1/packed/1±
LSTM/lstm/zeros_1/packedPack LSTM/lstm/strided_slice:output:0#LSTM/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM/lstm/zeros_1/packedw
LSTM/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/zeros_1/Const¶
LSTM/lstm/zeros_1Fill!LSTM/lstm/zeros_1/packed:output:0 LSTM/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/zeros_1Й
LSTM/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM/lstm/transpose/permШ
LSTM/lstm/transpose	Transposeinputs!LSTM/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:d€€€€€€€€€2
LSTM/lstm/transposem
LSTM/lstm/Shape_1ShapeLSTM/lstm/transpose:y:0*
T0*
_output_shapes
:2
LSTM/lstm/Shape_1М
LSTM/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
LSTM/lstm/strided_slice_1/stackР
!LSTM/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_1/stack_1Р
!LSTM/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_1/stack_2™
LSTM/lstm/strided_slice_1StridedSliceLSTM/lstm/Shape_1:output:0(LSTM/lstm/strided_slice_1/stack:output:0*LSTM/lstm/strided_slice_1/stack_1:output:0*LSTM/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM/lstm/strided_slice_1Щ
%LSTM/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2'
%LSTM/lstm/TensorArrayV2/element_shapeЏ
LSTM/lstm/TensorArrayV2TensorListReserve.LSTM/lstm/TensorArrayV2/element_shape:output:0"LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM/lstm/TensorArrayV2”
?LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2A
?LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape†
1LSTM/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorLSTM/lstm/transpose:y:0HLSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1LSTM/lstm/TensorArrayUnstack/TensorListFromTensorМ
LSTM/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
LSTM/lstm/strided_slice_2/stackР
!LSTM/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_2/stack_1Р
!LSTM/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_2/stack_2Є
LSTM/lstm/strided_slice_2StridedSliceLSTM/lstm/transpose:y:0(LSTM/lstm/strided_slice_2/stack:output:0*LSTM/lstm/strided_slice_2/stack_1:output:0*LSTM/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
LSTM/lstm/strided_slice_2 
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp2lstm_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02+
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOpћ
LSTM/lstm/lstm_cell/MatMulMatMul"LSTM/lstm/strided_slice_2:output:01LSTM/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/MatMul—
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp»
LSTM/lstm/lstm_cell/MatMul_1MatMulLSTM/lstm/zeros:output:03LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/MatMul_1Љ
LSTM/lstm/lstm_cell/addAddV2$LSTM/lstm/lstm_cell/MatMul:product:0&LSTM/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/add…
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp3lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp…
LSTM/lstm/lstm_cell/BiasAddBiasAddLSTM/lstm/lstm_cell/add:z:02LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/BiasAddx
LSTM/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/lstm_cell/ConstМ
#LSTM/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#LSTM/lstm/lstm_cell/split/split_dimУ
LSTM/lstm/lstm_cell/splitSplit,LSTM/lstm/lstm_cell/split/split_dim:output:0$LSTM/lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
LSTM/lstm/lstm_cell/splitУ
LSTM/lstm/lstm_cell/TanhTanh"LSTM/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/TanhЧ
LSTM/lstm/lstm_cell/Tanh_1Tanh"LSTM/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/Tanh_1®
LSTM/lstm/lstm_cell/mulMulLSTM/lstm/lstm_cell/Tanh_1:y:0LSTM/lstm/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/mulЧ
LSTM/lstm/lstm_cell/Tanh_2Tanh"LSTM/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/Tanh_2Ѓ
LSTM/lstm/lstm_cell/mul_1MulLSTM/lstm/lstm_cell/Tanh:y:0LSTM/lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/mul_1Ѓ
LSTM/lstm/lstm_cell/add_1AddV2LSTM/lstm/lstm_cell/mul:z:0LSTM/lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/add_1Ч
LSTM/lstm/lstm_cell/Tanh_3Tanh"LSTM/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/Tanh_3Т
LSTM/lstm/lstm_cell/Tanh_4TanhLSTM/lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/Tanh_4∞
LSTM/lstm/lstm_cell/mul_2MulLSTM/lstm/lstm_cell/Tanh_3:y:0LSTM/lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/mul_2£
'LSTM/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2)
'LSTM/lstm/TensorArrayV2_1/element_shapeа
LSTM/lstm/TensorArrayV2_1TensorListReserve0LSTM/lstm/TensorArrayV2_1/element_shape:output:0"LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM/lstm/TensorArrayV2_1b
LSTM/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM/lstm/timeУ
"LSTM/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2$
"LSTM/lstm/while/maximum_iterations~
LSTM/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM/lstm/while/loop_counterГ
LSTM/lstm/whileWhile%LSTM/lstm/while/loop_counter:output:0+LSTM/lstm/while/maximum_iterations:output:0LSTM/lstm/time:output:0"LSTM/lstm/TensorArrayV2_1:handle:0LSTM/lstm/zeros:output:0LSTM/lstm/zeros_1:output:0"LSTM/lstm/strided_slice_1:output:0ALSTM/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_lstm_lstm_cell_matmul_readvariableop_resource4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource3lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*)
body!R
LSTM_lstm_while_body_11856827*)
cond!R
LSTM_lstm_while_cond_11856826*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
LSTM/lstm/while…
:LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2<
:LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shapeС
,LSTM/lstm/TensorArrayV2Stack/TensorListStackTensorListStackLSTM/lstm/while:output:3CLSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d€€€€€€€€€А*
element_dtype02.
,LSTM/lstm/TensorArrayV2Stack/TensorListStackХ
LSTM/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2!
LSTM/lstm/strided_slice_3/stackР
!LSTM/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!LSTM/lstm/strided_slice_3/stack_1Р
!LSTM/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_3/stack_2„
LSTM/lstm/strided_slice_3StridedSlice5LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0(LSTM/lstm/strided_slice_3/stack:output:0*LSTM/lstm/strided_slice_3/stack_1:output:0*LSTM/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
LSTM/lstm/strided_slice_3Н
LSTM/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM/lstm/transpose_1/permќ
LSTM/lstm/transpose_1	Transpose5LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0#LSTM/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€dА2
LSTM/lstm/transpose_1z
LSTM/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/runtimeµ
(tf.math.l2_normalize/l2_normalize/SquareSquare"LSTM/lstm/strided_slice_3:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(tf.math.l2_normalize/l2_normalize/Squareі
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesИ
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/SumЯ
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *€жџ.2-
+tf.math.l2_normalize/l2_normalize/Maximum/yщ
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2+
)tf.math.l2_normalize/l2_normalize/MaximumЉ
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2)
'tf.math.l2_normalize/l2_normalize/Rsqrt—
!tf.math.l2_normalize/l2_normalizeMul"LSTM/lstm/strided_slice_3:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!tf.math.l2_normalize/l2_normalizeр
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulЗ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulЫ
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0+^LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp*^LSTM/lstm/lstm_cell/MatMul/ReadVariableOp,^LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^LSTM/lstm/while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2X
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp2V
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp2Z
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2"
LSTM/lstm/whileLSTM/lstm/while:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
Ж
М
'__inference_LSTM_layer_call_fn_11856216
input_1
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_LSTM_layer_call_and_return_conditional_losses_118562072
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€d
!
_user_specified_name	input_1
є
Ќ
while_cond_11855852
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_11855852___redundant_placeholder06
2while_while_cond_11855852___redundant_placeholder16
2while_while_cond_11855852___redundant_placeholder26
2while_while_cond_11855852___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
ђX
е
#model_LSTM_lstm_while_body_11855008<
8model_lstm_lstm_while_model_lstm_lstm_while_loop_counterB
>model_lstm_lstm_while_model_lstm_lstm_while_maximum_iterations%
!model_lstm_lstm_while_placeholder'
#model_lstm_lstm_while_placeholder_1'
#model_lstm_lstm_while_placeholder_2'
#model_lstm_lstm_while_placeholder_3;
7model_lstm_lstm_while_model_lstm_lstm_strided_slice_1_0w
smodel_lstm_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0D
@model_lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0F
Bmodel_lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0E
Amodel_lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"
model_lstm_lstm_while_identity$
 model_lstm_lstm_while_identity_1$
 model_lstm_lstm_while_identity_2$
 model_lstm_lstm_while_identity_3$
 model_lstm_lstm_while_identity_4$
 model_lstm_lstm_while_identity_59
5model_lstm_lstm_while_model_lstm_lstm_strided_slice_1u
qmodel_lstm_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensorB
>model_lstm_lstm_while_lstm_cell_matmul_readvariableop_resourceD
@model_lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resourceC
?model_lstm_lstm_while_lstm_cell_biasadd_readvariableop_resourceИҐ6model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpҐ5model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpҐ7model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOpг
Gmodel/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gmodel/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape≥
9model/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsmodel_lstm_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0!model_lstm_lstm_while_placeholderPmodel/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02;
9model/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItemр
5model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@model_lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype027
5model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpО
&model/LSTM/lstm/while/lstm_cell/MatMulMatMul@model/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&model/LSTM/lstm/while/lstm_cell/MatMulч
7model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBmodel_lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype029
7model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOpч
(model/LSTM/lstm/while/lstm_cell/MatMul_1MatMul#model_lstm_lstm_while_placeholder_2?model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(model/LSTM/lstm/while/lstm_cell/MatMul_1м
#model/LSTM/lstm/while/lstm_cell/addAddV20model/LSTM/lstm/while/lstm_cell/MatMul:product:02model/LSTM/lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2%
#model/LSTM/lstm/while/lstm_cell/addп
6model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAmodel_lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype028
6model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpщ
'model/LSTM/lstm/while/lstm_cell/BiasAddBiasAdd'model/LSTM/lstm/while/lstm_cell/add:z:0>model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2)
'model/LSTM/lstm/while/lstm_cell/BiasAddР
%model/LSTM/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%model/LSTM/lstm/while/lstm_cell/Const§
/model/LSTM/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/model/LSTM/lstm/while/lstm_cell/split/split_dim√
%model/LSTM/lstm/while/lstm_cell/splitSplit8model/LSTM/lstm/while/lstm_cell/split/split_dim:output:00model/LSTM/lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2'
%model/LSTM/lstm/while/lstm_cell/splitЈ
$model/LSTM/lstm/while/lstm_cell/TanhTanh.model/LSTM/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2&
$model/LSTM/lstm/while/lstm_cell/Tanhї
&model/LSTM/lstm/while/lstm_cell/Tanh_1Tanh.model/LSTM/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2(
&model/LSTM/lstm/while/lstm_cell/Tanh_1’
#model/LSTM/lstm/while/lstm_cell/mulMul*model/LSTM/lstm/while/lstm_cell/Tanh_1:y:0#model_lstm_lstm_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2%
#model/LSTM/lstm/while/lstm_cell/mulї
&model/LSTM/lstm/while/lstm_cell/Tanh_2Tanh.model/LSTM/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2(
&model/LSTM/lstm/while/lstm_cell/Tanh_2ё
%model/LSTM/lstm/while/lstm_cell/mul_1Mul(model/LSTM/lstm/while/lstm_cell/Tanh:y:0*model/LSTM/lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%model/LSTM/lstm/while/lstm_cell/mul_1ё
%model/LSTM/lstm/while/lstm_cell/add_1AddV2'model/LSTM/lstm/while/lstm_cell/mul:z:0)model/LSTM/lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%model/LSTM/lstm/while/lstm_cell/add_1ї
&model/LSTM/lstm/while/lstm_cell/Tanh_3Tanh.model/LSTM/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2(
&model/LSTM/lstm/while/lstm_cell/Tanh_3ґ
&model/LSTM/lstm/while/lstm_cell/Tanh_4Tanh)model/LSTM/lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&model/LSTM/lstm/while/lstm_cell/Tanh_4а
%model/LSTM/lstm/while/lstm_cell/mul_2Mul*model/LSTM/lstm/while/lstm_cell/Tanh_3:y:0*model/LSTM/lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%model/LSTM/lstm/while/lstm_cell/mul_2≠
:model/LSTM/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#model_lstm_lstm_while_placeholder_1!model_lstm_lstm_while_placeholder)model/LSTM/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:model/LSTM/lstm/while/TensorArrayV2Write/TensorListSetItem|
model/LSTM/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/LSTM/lstm/while/add/y©
model/LSTM/lstm/while/addAddV2!model_lstm_lstm_while_placeholder$model/LSTM/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/while/addА
model/LSTM/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/LSTM/lstm/while/add_1/y∆
model/LSTM/lstm/while/add_1AddV28model_lstm_lstm_while_model_lstm_lstm_while_loop_counter&model/LSTM/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/while/add_1є
model/LSTM/lstm/while/IdentityIdentitymodel/LSTM/lstm/while/add_1:z:07^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
model/LSTM/lstm/while/Identity№
 model/LSTM/lstm/while/Identity_1Identity>model_lstm_lstm_while_model_lstm_lstm_while_maximum_iterations7^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model/LSTM/lstm/while/Identity_1ї
 model/LSTM/lstm/while/Identity_2Identitymodel/LSTM/lstm/while/add:z:07^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model/LSTM/lstm/while/Identity_2и
 model/LSTM/lstm/while/Identity_3IdentityJmodel/LSTM/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model/LSTM/lstm/while/Identity_3ў
 model/LSTM/lstm/while/Identity_4Identity)model/LSTM/lstm/while/lstm_cell/mul_2:z:07^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2"
 model/LSTM/lstm/while/Identity_4ў
 model/LSTM/lstm/while/Identity_5Identity)model/LSTM/lstm/while/lstm_cell/add_1:z:07^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2"
 model/LSTM/lstm/while/Identity_5"I
model_lstm_lstm_while_identity'model/LSTM/lstm/while/Identity:output:0"M
 model_lstm_lstm_while_identity_1)model/LSTM/lstm/while/Identity_1:output:0"M
 model_lstm_lstm_while_identity_2)model/LSTM/lstm/while/Identity_2:output:0"M
 model_lstm_lstm_while_identity_3)model/LSTM/lstm/while/Identity_3:output:0"M
 model_lstm_lstm_while_identity_4)model/LSTM/lstm/while/Identity_4:output:0"M
 model_lstm_lstm_while_identity_5)model/LSTM/lstm/while/Identity_5:output:0"Д
?model_lstm_lstm_while_lstm_cell_biasadd_readvariableop_resourceAmodel_lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"Ж
@model_lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resourceBmodel_lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"В
>model_lstm_lstm_while_lstm_cell_matmul_readvariableop_resource@model_lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0"p
5model_lstm_lstm_while_model_lstm_lstm_strided_slice_17model_lstm_lstm_while_model_lstm_lstm_strided_slice_1_0"и
qmodel_lstm_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensorsmodel_lstm_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2p
6model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp2n
5model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp5model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2r
7model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp7model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
є
Ќ
while_cond_11857748
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_11857748___redundant_placeholder06
2while_while_cond_11857748___redundant_placeholder16
2while_while_cond_11857748___redundant_placeholder26
2while_while_cond_11857748___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
µ@
е
while_body_11857749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceИҐ&while/lstm_cell/BiasAdd/ReadVariableOpҐ%while/lstm_cell/MatMul/ReadVariableOpҐ'while/lstm_cell/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemј
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpќ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul«
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpЈ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul_1ђ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/addњ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpє
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/ConstД
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimГ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
while/lstm_cell/splitЗ
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/TanhЛ
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_1Х
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mulЛ
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_2Ю
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_1Ю
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/add_1Л
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_3Ж
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_4†
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_2Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ў
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityм
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1џ
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2И
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3щ
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_4щ
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
я 
¬
B__inference_LSTM_layer_call_and_return_conditional_losses_11856181
input_1
lstm_11856161
lstm_11856163
lstm_11856165
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐlstm/StatefulPartitionedCallЩ
lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_11856161lstm_11856163lstm_11856165*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_layer_call_and_return_conditional_losses_118561152
lstm/StatefulPartitionedCallЋ
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856161*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulа
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856163* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul°
IdentityIdentity%lstm/StatefulPartitionedCall:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^lstm/StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€d
!
_user_specified_name	input_1
ж

Х
LSTM_lstm_while_cond_118568260
,lstm_lstm_while_lstm_lstm_while_loop_counter6
2lstm_lstm_while_lstm_lstm_while_maximum_iterations
lstm_lstm_while_placeholder!
lstm_lstm_while_placeholder_1!
lstm_lstm_while_placeholder_2!
lstm_lstm_while_placeholder_32
.lstm_lstm_while_less_lstm_lstm_strided_slice_1J
Flstm_lstm_while_lstm_lstm_while_cond_11856826___redundant_placeholder0J
Flstm_lstm_while_lstm_lstm_while_cond_11856826___redundant_placeholder1J
Flstm_lstm_while_lstm_lstm_while_cond_11856826___redundant_placeholder2J
Flstm_lstm_while_lstm_lstm_while_cond_11856826___redundant_placeholder3
lstm_lstm_while_identity
Ґ
LSTM/lstm/while/LessLesslstm_lstm_while_placeholder.lstm_lstm_while_less_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LSTM/lstm/while/Less{
LSTM/lstm/while/IdentityIdentityLSTM/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2
LSTM/lstm/while/Identity"=
lstm_lstm_while_identity!LSTM/lstm/while/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
ыr
й
B__inference_lstm_layer_call_and_return_conditional_losses_11858011
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ lstm_cell/BiasAdd/ReadVariableOpҐlstm_cell/MatMul/ReadVariableOpҐ!lstm_cell/MatMul_1/ReadVariableOpҐwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ђ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02!
lstm_cell/MatMul/ReadVariableOp§
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul≥
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp†
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul_1Ф
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/addЂ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp°
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimл
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_1А
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_2Ж
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_1Ж
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_4И
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterн
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_11857914*
condR
while_cond_11857913*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   22
0TensorArrayV2Stack/TensorListStack/element_shapeт
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permѓ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeж
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulэ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulж
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ж
Л
&__inference_signature_wrapper_11856587
input_1
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_118551002
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€d
!
_user_specified_name	input_1
О,
¬
C__inference_model_layer_call_and_return_conditional_losses_11856549

inputs
lstm_11856522
lstm_11856524
lstm_11856526
identityИҐLSTM/StatefulPartitionedCallҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpШ
LSTM/StatefulPartitionedCallStatefulPartitionedCallinputslstm_11856522lstm_11856524lstm_11856526*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_LSTM_layer_call_and_return_conditional_losses_118562072
LSTM/StatefulPartitionedCallЄ
(tf.math.l2_normalize/l2_normalize/SquareSquare%LSTM/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(tf.math.l2_normalize/l2_normalize/Squareі
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesИ
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/SumЯ
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *€жџ.2-
+tf.math.l2_normalize/l2_normalize/Maximum/yщ
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2+
)tf.math.l2_normalize/l2_normalize/MaximumЉ
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2)
'tf.math.l2_normalize/l2_normalize/Rsqrt‘
!tf.math.l2_normalize/l2_normalizeMul%LSTM/StatefulPartitionedCall:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!tf.math.l2_normalize/l2_normalizeЋ
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856522*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulа
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856524* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul°
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0^LSTM/StatefulPartitionedCall=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2<
LSTM/StatefulPartitionedCallLSTM/StatefulPartitionedCall2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
√
ћ
,__inference_lstm_cell_layer_call_fn_11858169

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ИҐStatefulPartitionedCall≈
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_118552302
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

IdentityУ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1У

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€А:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
states/0:RN
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
states/1
Г
Л
'__inference_LSTM_layer_call_fn_11857306

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_LSTM_layer_call_and_return_conditional_losses_118562072
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
ј7
в
G__inference_lstm_cell_layer_call_and_return_conditional_losses_11855230

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim√
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
splitW
TanhTanhsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh[
Tanh_1Tanhsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_1Z
mulMul
Tanh_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€А2
mul[
Tanh_2Tanhsplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_2^
mul_1MulTanh:y:0
Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
add_1[
Tanh_3Tanhsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_3V
Tanh_4Tanh	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_4`
mul_2Mul
Tanh_3:y:0
Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mul_2№
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulу
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul±
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityµ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1µ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_namestates:PL
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_namestates
»7
д
G__inference_lstm_cell_layer_call_and_return_conditional_losses_11858090

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim√
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
splitW
TanhTanhsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh[
Tanh_1Tanhsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_1Z
mulMul
Tanh_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€А2
mul[
Tanh_2Tanhsplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_2^
mul_1MulTanh:y:0
Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
add_1[
Tanh_3Tanhsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_3V
Tanh_4Tanh	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tanh_4`
mul_2Mul
Tanh_3:y:0
Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mul_2№
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulу
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul±
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityµ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1µ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
states/0:RN
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
states/1
№]
Ё
B__inference_lstm_layer_call_and_return_conditional_losses_11855629

inputs
lstm_cell_11855535
lstm_cell_11855537
lstm_cell_11855539
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ!lstm_cell/StatefulPartitionedCallҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2Ц
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_11855535lstm_cell_11855537lstm_cell_11855539*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_118551852#
!lstm_cell/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter®
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_11855535lstm_cell_11855537lstm_cell_11855539*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_11855548*
condR
while_cond_11855547*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   22
0TensorArrayV2Stack/TensorListStack/element_shapeт
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permѓ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime–
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_11855535*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulе
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_11855537* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul°
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
С,
√
C__inference_model_layer_call_and_return_conditional_losses_11856475
input_1
lstm_11856448
lstm_11856450
lstm_11856452
identityИҐLSTM/StatefulPartitionedCallҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЩ
LSTM/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_11856448lstm_11856450lstm_11856452*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_LSTM_layer_call_and_return_conditional_losses_118562072
LSTM/StatefulPartitionedCallЄ
(tf.math.l2_normalize/l2_normalize/SquareSquare%LSTM/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(tf.math.l2_normalize/l2_normalize/Squareі
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesИ
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/SumЯ
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *€жџ.2-
+tf.math.l2_normalize/l2_normalize/Maximum/yщ
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2+
)tf.math.l2_normalize/l2_normalize/MaximumЉ
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2)
'tf.math.l2_normalize/l2_normalize/Rsqrt‘
!tf.math.l2_normalize/l2_normalizeMul%LSTM/StatefulPartitionedCall:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!tf.math.l2_normalize/l2_normalizeЋ
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856448*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulа
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856450* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul°
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0^LSTM/StatefulPartitionedCall=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2<
LSTM/StatefulPartitionedCallLSTM/StatefulPartitionedCall2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:€€€€€€€€€d
!
_user_specified_name	input_1
ш3
г
$__inference__traced_restore_11858290
file_prefix!
assignvariableop_adagrad_iter$
 assignvariableop_1_adagrad_decay,
(assignvariableop_2_adagrad_learning_rate1
-assignvariableop_3_lstm_lstm_lstm_cell_kernel;
7assignvariableop_4_lstm_lstm_lstm_cell_recurrent_kernel/
+assignvariableop_5_lstm_lstm_lstm_cell_bias
assignvariableop_6_total
assignvariableop_7_countE
Aassignvariableop_8_adagrad_lstm_lstm_lstm_cell_kernel_accumulatorO
Kassignvariableop_9_adagrad_lstm_lstm_lstm_cell_recurrent_kernel_accumulatorD
@assignvariableop_10_adagrad_lstm_lstm_lstm_cell_bias_accumulator
identity_12ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Љ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*»
valueЊBїB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¶
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesз
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

IdentityЬ
AssignVariableOpAssignVariableOpassignvariableop_adagrad_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1•
AssignVariableOp_1AssignVariableOp assignvariableop_1_adagrad_decayIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2≠
AssignVariableOp_2AssignVariableOp(assignvariableop_2_adagrad_learning_rateIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3≤
AssignVariableOp_3AssignVariableOp-assignvariableop_3_lstm_lstm_lstm_cell_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Љ
AssignVariableOp_4AssignVariableOp7assignvariableop_4_lstm_lstm_lstm_cell_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5∞
AssignVariableOp_5AssignVariableOp+assignvariableop_5_lstm_lstm_lstm_cell_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Э
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Э
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8∆
AssignVariableOp_8AssignVariableOpAassignvariableop_8_adagrad_lstm_lstm_lstm_cell_kernel_accumulatorIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9–
AssignVariableOp_9AssignVariableOpKassignvariableop_9_adagrad_lstm_lstm_lstm_cell_recurrent_kernel_accumulatorIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10»
AssignVariableOp_10AssignVariableOp@assignvariableop_10_adagrad_lstm_lstm_lstm_cell_bias_accumulatorIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp–
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_11√
Identity_12IdentityIdentity_11:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_12"#
identity_12Identity_12:output:0*A
_input_shapes0
.: :::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
є
Ќ
while_cond_11857561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_11857561___redundant_placeholder06
2while_while_cond_11857561___redundant_placeholder16
2while_while_cond_11857561___redundant_placeholder26
2while_while_cond_11857561___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
И
Н
(__inference_model_layer_call_fn_11856558
input_1
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_118565492
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€d
!
_user_specified_name	input_1
¶|
К
B__inference_LSTM_layer_call_and_return_conditional_losses_11857130

inputs1
-lstm_lstm_cell_matmul_readvariableop_resource3
/lstm_lstm_cell_matmul_1_readvariableop_resource2
.lstm_lstm_cell_biasadd_readvariableop_resource
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ%lstm/lstm_cell/BiasAdd/ReadVariableOpҐ$lstm/lstm_cell/MatMul/ReadVariableOpҐ&lstm/lstm_cell/MatMul_1/ReadVariableOpҐ
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stackВ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1В
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2А
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_sliceg
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros/mul/yА
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessm
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros/packed/1Ч
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/ConstК

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

lstm/zerosk
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros_1/mul/yЖ
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm/zeros_1/Less/yГ
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessq
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
lstm/zeros_1/packed/1Э
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/ConstТ
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/permЙ
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:d€€€€€€€€€2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1В
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stackЖ
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1Ж
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2М
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1П
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2"
 lstm/TensorArrayV2/element_shape∆
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2…
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensorВ
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stackЖ
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1Ж
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2Ъ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm/strided_slice_2ї
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOpЄ
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/MatMul¬
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpі
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/MatMul_1®
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/addЇ
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOpµ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/BiasAddn
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/ConstВ
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim€
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm/lstm_cell/splitД
lstm/lstm_cell/TanhTanhlstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/TanhИ
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_1Ф
lstm/lstm_cell/mulMullstm/lstm_cell/Tanh_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/mulИ
lstm/lstm_cell/Tanh_2Tanhlstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_2Ъ
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Tanh:y:0lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/mul_1Ъ
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/add_1И
lstm/lstm_cell/Tanh_3Tanhlstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_3Г
lstm/lstm_cell/Tanh_4Tanhlstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/Tanh_4Ь
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Tanh_3:y:0lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/lstm_cell/mul_2Щ
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2$
"lstm/TensorArrayV2_1/element_shapeћ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/timeЙ
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterЄ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_while_body_11857033*$
condR
lstm_while_cond_11857032*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2

lstm/whileњ
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeэ
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d€€€€€€€€€А*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStackЛ
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm/strided_slice_3/stackЖ
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1Ж
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2є
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
lstm/strided_slice_3Г
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permЇ
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€dА2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeл
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulВ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul€
IdentityIdentitylstm/strided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
≈r
з
B__inference_lstm_layer_call_and_return_conditional_losses_11857494

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ lstm_cell/BiasAdd/ReadVariableOpҐlstm_cell/MatMul/ReadVariableOpҐ!lstm_cell/MatMul_1/ReadVariableOpҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d€€€€€€€€€2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ђ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02!
lstm_cell/MatMul/ReadVariableOp§
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul≥
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp†
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul_1Ф
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/addЂ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp°
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimл
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_1А
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_2Ж
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_1Ж
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_4И
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterн
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_11857397*
condR
while_cond_11857396*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d€€€€€€€€€А*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€dА2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeж
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulэ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulж
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
№ 
Ѕ
B__inference_LSTM_layer_call_and_return_conditional_losses_11856207

inputs
lstm_11856187
lstm_11856189
lstm_11856191
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐlstm/StatefulPartitionedCallШ
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_11856187lstm_11856189lstm_11856191*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_layer_call_and_return_conditional_losses_118561152
lstm/StatefulPartitionedCallЋ
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856187*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulа
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_11856189* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul°
IdentityIdentity%lstm/StatefulPartitionedCall:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^lstm/StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
µ@
е
while_body_11857397
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceИҐ&while/lstm_cell/BiasAdd/ReadVariableOpҐ%while/lstm_cell/MatMul/ReadVariableOpҐ'while/lstm_cell/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemј
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpќ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul«
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpЈ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul_1ђ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/addњ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpє
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/ConstД
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimГ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
while/lstm_cell/splitЗ
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/TanhЛ
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_1Х
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mulЛ
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_2Ю
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_1Ю
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/add_1Л
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_3Ж
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_4†
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_2Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ў
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityм
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1џ
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2И
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3щ
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_4щ
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
Є
ї
__inference_loss_fn_0_11858180I
Elstm_lstm_lstm_cell_kernel_regularizer_square_readvariableop_resource
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpГ
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpElstm_lstm_lstm_cell_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul∞
IdentityIdentity.LSTM/lstm/lstm_cell/kernel/Regularizer/mul:z:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp
м$
К
while_body_11855548
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_11855572_0
while_lstm_cell_11855574_0
while_lstm_cell_11855576_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_11855572
while_lstm_cell_11855574
while_lstm_cell_11855576ИҐ'while/lstm_cell/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЏ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_11855572_0while_lstm_cell_11855574_0while_lstm_cell_11855576_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_118551852)
'while/lstm_cell/StatefulPartitionedCallф
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1И
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/IdentityЫ
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1К
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Ј
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3њ
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_4њ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_11855572while_lstm_cell_11855572_0"6
while_lstm_cell_11855574while_lstm_cell_11855574_0"6
while_lstm_cell_11855576while_lstm_cell_11855576_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
Ђ
ѕ
__inference_loss_fn_1_11858191S
Olstm_lstm_lstm_cell_recurrent_kernel_regularizer_square_readvariableop_resource
identityИҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpOlstm_lstm_lstm_cell_recurrent_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulƒ
IdentityIdentity8LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0G^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
µ@
е
while_body_11856018
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceИҐ&while/lstm_cell/BiasAdd/ReadVariableOpҐ%while/lstm_cell/MatMul/ReadVariableOpҐ'while/lstm_cell/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemј
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpќ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul«
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpЈ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul_1ђ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/addњ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpє
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/ConstД
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimГ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
while/lstm_cell/splitЗ
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/TanhЛ
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_1Х
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mulЛ
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_2Ю
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_1Ю
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/add_1Л
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_3Ж
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_4†
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_2Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ў
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityм
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1џ
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2И
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3щ
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_4щ
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
сG
Е	
lstm_while_body_11856298&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_09
5lstm_while_lstm_cell_matmul_readvariableop_resource_0;
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor7
3lstm_while_lstm_cell_matmul_readvariableop_resource9
5lstm_while_lstm_cell_matmul_1_readvariableop_resource8
4lstm_while_lstm_cell_biasadd_readvariableop_resourceИҐ+lstm/while/lstm_cell/BiasAdd/ReadVariableOpҐ*lstm/while/lstm_cell/MatMul/ReadVariableOpҐ,lstm/while/lstm_cell/MatMul_1/ReadVariableOpЌ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeс
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemѕ
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpв
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/MatMul÷
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpЋ
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/MatMul_1ј
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/addќ
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpЌ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/BiasAddz
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/ConstО
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimЧ
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm/while/lstm_cell/splitЦ
lstm/while/lstm_cell/TanhTanh#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/TanhЪ
lstm/while/lstm_cell/Tanh_1Tanh#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_1©
lstm/while/lstm_cell/mulMullstm/while/lstm_cell/Tanh_1:y:0lstm_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/mulЪ
lstm/while/lstm_cell/Tanh_2Tanh#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_2≤
lstm/while/lstm_cell/mul_1Mullstm/while/lstm_cell/Tanh:y:0lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/mul_1≤
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/add_1Ъ
lstm/while/lstm_cell/Tanh_3Tanh#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_3Х
lstm/while/lstm_cell/Tanh_4Tanhlstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_4і
lstm/while/lstm_cell/mul_2Mullstm/while/lstm_cell/Tanh_3:y:0lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/mul_2ц
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/yП
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1ч
lstm/while/IdentityIdentitylstm/while/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/IdentityП
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1щ
lstm/while/Identity_2Identitylstm/while/add:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2¶
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3Ч
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/Identity_4Ч
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"Љ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
є
Ќ
while_cond_11855691
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_11855691___redundant_placeholder06
2while_while_cond_11855691___redundant_placeholder16
2while_while_cond_11855691___redundant_placeholder26
2while_while_cond_11855691___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
–	
±
lstm_while_cond_11856297&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1@
<lstm_while_lstm_while_cond_11856297___redundant_placeholder0@
<lstm_while_lstm_while_cond_11856297___redundant_placeholder1@
<lstm_while_lstm_while_cond_11856297___redundant_placeholder2@
<lstm_while_lstm_while_cond_11856297___redundant_placeholder3
lstm_while_identity
Й
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
Ы
Н
#model_LSTM_lstm_while_cond_11855007<
8model_lstm_lstm_while_model_lstm_lstm_while_loop_counterB
>model_lstm_lstm_while_model_lstm_lstm_while_maximum_iterations%
!model_lstm_lstm_while_placeholder'
#model_lstm_lstm_while_placeholder_1'
#model_lstm_lstm_while_placeholder_2'
#model_lstm_lstm_while_placeholder_3>
:model_lstm_lstm_while_less_model_lstm_lstm_strided_slice_1V
Rmodel_lstm_lstm_while_model_lstm_lstm_while_cond_11855007___redundant_placeholder0V
Rmodel_lstm_lstm_while_model_lstm_lstm_while_cond_11855007___redundant_placeholder1V
Rmodel_lstm_lstm_while_model_lstm_lstm_while_cond_11855007___redundant_placeholder2V
Rmodel_lstm_lstm_while_model_lstm_lstm_while_cond_11855007___redundant_placeholder3"
model_lstm_lstm_while_identity
ј
model/LSTM/lstm/while/LessLess!model_lstm_lstm_while_placeholder:model_lstm_lstm_while_less_model_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
model/LSTM/lstm/while/LessН
model/LSTM/lstm/while/IdentityIdentitymodel/LSTM/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2 
model/LSTM/lstm/while/Identity"I
model_lstm_lstm_while_identity'model/LSTM/lstm/while/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
є
Ќ
while_cond_11857396
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_11857396___redundant_placeholder06
2while_while_cond_11857396___redundant_placeholder16
2while_while_cond_11857396___redundant_placeholder26
2while_while_cond_11857396___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
Ж
М
'__inference_LSTM_layer_call_fn_11856227
input_1
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_LSTM_layer_call_and_return_conditional_losses_118562072
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€d
!
_user_specified_name	input_1
Г
Л
'__inference_lstm_layer_call_fn_11857670

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_layer_call_and_return_conditional_losses_118559502
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
є
Ќ
while_cond_11855547
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_11855547___redundant_placeholder06
2while_while_cond_11855547___redundant_placeholder16
2while_while_cond_11855547___redundant_placeholder26
2while_while_cond_11855547___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
≈r
з
B__inference_lstm_layer_call_and_return_conditional_losses_11855950

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ lstm_cell/BiasAdd/ReadVariableOpҐlstm_cell/MatMul/ReadVariableOpҐ!lstm_cell/MatMul_1/ReadVariableOpҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d€€€€€€€€€2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ђ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02!
lstm_cell/MatMul/ReadVariableOp§
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul≥
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp†
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul_1Ф
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/addЂ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp°
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimл
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_1А
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_2Ж
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_1Ж
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_4И
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterн
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_11855853*
condR
while_cond_11855852*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d€€€€€€€€€А*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€dА2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeж
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulэ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulж
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
сG
Е	
lstm_while_body_11857033&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_09
5lstm_while_lstm_cell_matmul_readvariableop_resource_0;
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor7
3lstm_while_lstm_cell_matmul_readvariableop_resource9
5lstm_while_lstm_cell_matmul_1_readvariableop_resource8
4lstm_while_lstm_cell_biasadd_readvariableop_resourceИҐ+lstm/while/lstm_cell/BiasAdd/ReadVariableOpҐ*lstm/while/lstm_cell/MatMul/ReadVariableOpҐ,lstm/while/lstm_cell/MatMul_1/ReadVariableOpЌ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeс
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemѕ
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpв
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/MatMul÷
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpЋ
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/MatMul_1ј
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/addќ
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpЌ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/BiasAddz
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/ConstО
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimЧ
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm/while/lstm_cell/splitЦ
lstm/while/lstm_cell/TanhTanh#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/TanhЪ
lstm/while/lstm_cell/Tanh_1Tanh#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_1©
lstm/while/lstm_cell/mulMullstm/while/lstm_cell/Tanh_1:y:0lstm_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/mulЪ
lstm/while/lstm_cell/Tanh_2Tanh#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_2≤
lstm/while/lstm_cell/mul_1Mullstm/while/lstm_cell/Tanh:y:0lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/mul_1≤
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/add_1Ъ
lstm/while/lstm_cell/Tanh_3Tanh#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_3Х
lstm/while/lstm_cell/Tanh_4Tanhlstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/Tanh_4і
lstm/while/lstm_cell/mul_2Mullstm/while/lstm_cell/Tanh_3:y:0lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/lstm_cell/mul_2ц
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/yП
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1ч
lstm/while/IdentityIdentitylstm/while/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/IdentityП
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1щ
lstm/while/Identity_2Identitylstm/while/add:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2¶
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3Ч
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/Identity_4Ч
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"Љ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
є
Ќ
while_cond_11856017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_11856017___redundant_placeholder06
2while_while_cond_11856017___redundant_placeholder16
2while_while_cond_11856017___redundant_placeholder26
2while_while_cond_11856017___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :€€€€€€€€€А:€€€€€€€€€А: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
:
Г
Л
'__inference_lstm_layer_call_fn_11857681

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_layer_call_and_return_conditional_losses_118561152
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
Ы
Н
'__inference_lstm_layer_call_fn_11858033
inputs_0
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_layer_call_and_return_conditional_losses_118557732
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
√
ћ
,__inference_lstm_cell_layer_call_fn_11858152

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ИҐStatefulPartitionedCall≈
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_118551852
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

IdentityУ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1У

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€А:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
states/0:RN
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
states/1
ыr
й
B__inference_lstm_layer_call_and_return_conditional_losses_11857846
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityИҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ lstm_cell/BiasAdd/ReadVariableOpҐlstm_cell/MatMul/ReadVariableOpҐ!lstm_cell/MatMul_1/ReadVariableOpҐwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ђ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02!
lstm_cell/MatMul/ReadVariableOp§
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul≥
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp†
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/MatMul_1Ф
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/addЂ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp°
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimл
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_1А
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_2Ж
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_1Ж
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/Tanh_4И
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterн
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_11857749*
condR
while_cond_11857748*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   22
0TensorArrayV2Stack/TensorListStack/element_shapeт
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permѓ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeж
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulэ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulж
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
м$
К
while_body_11855692
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_11855716_0
while_lstm_cell_11855718_0
while_lstm_cell_11855720_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_11855716
while_lstm_cell_11855718
while_lstm_cell_11855720ИҐ'while/lstm_cell/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЏ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_11855716_0while_lstm_cell_11855718_0while_lstm_cell_11855720_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_118552302)
'while/lstm_cell/StatefulPartitionedCallф
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1И
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/IdentityЫ
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1К
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Ј
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3њ
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_4њ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_11855716while_lstm_cell_11855716_0"6
while_lstm_cell_11855718while_lstm_cell_11855718_0"6
while_lstm_cell_11855720while_lstm_cell_11855720_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
µ@
е
while_body_11857914
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceИҐ&while/lstm_cell/BiasAdd/ReadVariableOpҐ%while/lstm_cell/MatMul/ReadVariableOpҐ'while/lstm_cell/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemј
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpќ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul«
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpЈ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/MatMul_1ђ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/addњ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpє
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/ConstД
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimГ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
while/lstm_cell/splitЗ
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/TanhЛ
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_1Х
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mulЛ
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_2Ю
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_1Ю
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/add_1Л
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_3Ж
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/Tanh_4†
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell/mul_2Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ў
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityм
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1џ
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2И
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3щ
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_4щ
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :€€€€€€€€€А:€€€€€€€€€А: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:€€€€€€€€€А:.*
(
_output_shapes
:€€€€€€€€€А:

_output_shapes
: :

_output_shapes
: 
°С
Ѓ
C__inference_model_layer_call_and_return_conditional_losses_11856759

inputs6
2lstm_lstm_lstm_cell_matmul_readvariableop_resource8
4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource7
3lstm_lstm_lstm_cell_biasadd_readvariableop_resource
identityИҐ*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOpҐ)LSTM/lstm/lstm_cell/MatMul/ReadVariableOpҐ+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOpҐ<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐLSTM/lstm/whileX
LSTM/lstm/ShapeShapeinputs*
T0*
_output_shapes
:2
LSTM/lstm/ShapeИ
LSTM/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
LSTM/lstm/strided_slice/stackМ
LSTM/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
LSTM/lstm/strided_slice/stack_1М
LSTM/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
LSTM/lstm/strided_slice/stack_2Ю
LSTM/lstm/strided_sliceStridedSliceLSTM/lstm/Shape:output:0&LSTM/lstm/strided_slice/stack:output:0(LSTM/lstm/strided_slice/stack_1:output:0(LSTM/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM/lstm/strided_sliceq
LSTM/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
LSTM/lstm/zeros/mul/yФ
LSTM/lstm/zeros/mulMul LSTM/lstm/strided_slice:output:0LSTM/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros/muls
LSTM/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
LSTM/lstm/zeros/Less/yП
LSTM/lstm/zeros/LessLessLSTM/lstm/zeros/mul:z:0LSTM/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros/Lessw
LSTM/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
LSTM/lstm/zeros/packed/1Ђ
LSTM/lstm/zeros/packedPack LSTM/lstm/strided_slice:output:0!LSTM/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM/lstm/zeros/packeds
LSTM/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/zeros/ConstЮ
LSTM/lstm/zerosFillLSTM/lstm/zeros/packed:output:0LSTM/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/zerosu
LSTM/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
LSTM/lstm/zeros_1/mul/yЪ
LSTM/lstm/zeros_1/mulMul LSTM/lstm/strided_slice:output:0 LSTM/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros_1/mulw
LSTM/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
LSTM/lstm/zeros_1/Less/yЧ
LSTM/lstm/zeros_1/LessLessLSTM/lstm/zeros_1/mul:z:0!LSTM/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros_1/Less{
LSTM/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
LSTM/lstm/zeros_1/packed/1±
LSTM/lstm/zeros_1/packedPack LSTM/lstm/strided_slice:output:0#LSTM/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM/lstm/zeros_1/packedw
LSTM/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/zeros_1/Const¶
LSTM/lstm/zeros_1Fill!LSTM/lstm/zeros_1/packed:output:0 LSTM/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/zeros_1Й
LSTM/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM/lstm/transpose/permШ
LSTM/lstm/transpose	Transposeinputs!LSTM/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:d€€€€€€€€€2
LSTM/lstm/transposem
LSTM/lstm/Shape_1ShapeLSTM/lstm/transpose:y:0*
T0*
_output_shapes
:2
LSTM/lstm/Shape_1М
LSTM/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
LSTM/lstm/strided_slice_1/stackР
!LSTM/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_1/stack_1Р
!LSTM/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_1/stack_2™
LSTM/lstm/strided_slice_1StridedSliceLSTM/lstm/Shape_1:output:0(LSTM/lstm/strided_slice_1/stack:output:0*LSTM/lstm/strided_slice_1/stack_1:output:0*LSTM/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM/lstm/strided_slice_1Щ
%LSTM/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2'
%LSTM/lstm/TensorArrayV2/element_shapeЏ
LSTM/lstm/TensorArrayV2TensorListReserve.LSTM/lstm/TensorArrayV2/element_shape:output:0"LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM/lstm/TensorArrayV2”
?LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2A
?LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape†
1LSTM/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorLSTM/lstm/transpose:y:0HLSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1LSTM/lstm/TensorArrayUnstack/TensorListFromTensorМ
LSTM/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
LSTM/lstm/strided_slice_2/stackР
!LSTM/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_2/stack_1Р
!LSTM/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_2/stack_2Є
LSTM/lstm/strided_slice_2StridedSliceLSTM/lstm/transpose:y:0(LSTM/lstm/strided_slice_2/stack:output:0*LSTM/lstm/strided_slice_2/stack_1:output:0*LSTM/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
LSTM/lstm/strided_slice_2 
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp2lstm_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02+
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOpћ
LSTM/lstm/lstm_cell/MatMulMatMul"LSTM/lstm/strided_slice_2:output:01LSTM/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/MatMul—
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp»
LSTM/lstm/lstm_cell/MatMul_1MatMulLSTM/lstm/zeros:output:03LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/MatMul_1Љ
LSTM/lstm/lstm_cell/addAddV2$LSTM/lstm/lstm_cell/MatMul:product:0&LSTM/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/add…
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp3lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp…
LSTM/lstm/lstm_cell/BiasAddBiasAddLSTM/lstm/lstm_cell/add:z:02LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/BiasAddx
LSTM/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/lstm_cell/ConstМ
#LSTM/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#LSTM/lstm/lstm_cell/split/split_dimУ
LSTM/lstm/lstm_cell/splitSplit,LSTM/lstm/lstm_cell/split/split_dim:output:0$LSTM/lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
	num_split2
LSTM/lstm/lstm_cell/splitУ
LSTM/lstm/lstm_cell/TanhTanh"LSTM/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/TanhЧ
LSTM/lstm/lstm_cell/Tanh_1Tanh"LSTM/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/Tanh_1®
LSTM/lstm/lstm_cell/mulMulLSTM/lstm/lstm_cell/Tanh_1:y:0LSTM/lstm/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/mulЧ
LSTM/lstm/lstm_cell/Tanh_2Tanh"LSTM/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/Tanh_2Ѓ
LSTM/lstm/lstm_cell/mul_1MulLSTM/lstm/lstm_cell/Tanh:y:0LSTM/lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/mul_1Ѓ
LSTM/lstm/lstm_cell/add_1AddV2LSTM/lstm/lstm_cell/mul:z:0LSTM/lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/add_1Ч
LSTM/lstm/lstm_cell/Tanh_3Tanh"LSTM/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/Tanh_3Т
LSTM/lstm/lstm_cell/Tanh_4TanhLSTM/lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/Tanh_4∞
LSTM/lstm/lstm_cell/mul_2MulLSTM/lstm/lstm_cell/Tanh_3:y:0LSTM/lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
LSTM/lstm/lstm_cell/mul_2£
'LSTM/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2)
'LSTM/lstm/TensorArrayV2_1/element_shapeа
LSTM/lstm/TensorArrayV2_1TensorListReserve0LSTM/lstm/TensorArrayV2_1/element_shape:output:0"LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM/lstm/TensorArrayV2_1b
LSTM/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM/lstm/timeУ
"LSTM/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2$
"LSTM/lstm/while/maximum_iterations~
LSTM/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM/lstm/while/loop_counterГ
LSTM/lstm/whileWhile%LSTM/lstm/while/loop_counter:output:0+LSTM/lstm/while/maximum_iterations:output:0LSTM/lstm/time:output:0"LSTM/lstm/TensorArrayV2_1:handle:0LSTM/lstm/zeros:output:0LSTM/lstm/zeros_1:output:0"LSTM/lstm/strided_slice_1:output:0ALSTM/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_lstm_lstm_cell_matmul_readvariableop_resource4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource3lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *%
_read_only_resource_inputs
	
*)
body!R
LSTM_lstm_while_body_11856655*)
cond!R
LSTM_lstm_while_cond_11856654*M
output_shapes<
:: : : : :€€€€€€€€€А:€€€€€€€€€А: : : : : *
parallel_iterations 2
LSTM/lstm/while…
:LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2<
:LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shapeС
,LSTM/lstm/TensorArrayV2Stack/TensorListStackTensorListStackLSTM/lstm/while:output:3CLSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d€€€€€€€€€А*
element_dtype02.
,LSTM/lstm/TensorArrayV2Stack/TensorListStackХ
LSTM/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2!
LSTM/lstm/strided_slice_3/stackР
!LSTM/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!LSTM/lstm/strided_slice_3/stack_1Р
!LSTM/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_3/stack_2„
LSTM/lstm/strided_slice_3StridedSlice5LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0(LSTM/lstm/strided_slice_3/stack:output:0*LSTM/lstm/strided_slice_3/stack_1:output:0*LSTM/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_mask2
LSTM/lstm/strided_slice_3Н
LSTM/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM/lstm/transpose_1/permќ
LSTM/lstm/transpose_1	Transpose5LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0#LSTM/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€dА2
LSTM/lstm/transpose_1z
LSTM/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/runtimeµ
(tf.math.l2_normalize/l2_normalize/SquareSquare"LSTM/lstm/strided_slice_3:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(tf.math.l2_normalize/l2_normalize/Squareі
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesИ
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/SumЯ
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *€жџ.2-
+tf.math.l2_normalize/l2_normalize/Maximum/yщ
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2+
)tf.math.l2_normalize/l2_normalize/MaximumЉ
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2)
'tf.math.l2_normalize/l2_normalize/Rsqrt—
!tf.math.l2_normalize/l2_normalizeMul"LSTM/lstm/strided_slice_3:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!tf.math.l2_normalize/l2_normalizeр
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЎ
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square≠
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Constк
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum°
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xм
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulЗ
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpч
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareЅ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstТ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xФ
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulЫ
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0+^LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp*^LSTM/lstm/lstm_cell/MatMul/ReadVariableOp,^LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^LSTM/lstm/while*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€d:::2X
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp2V
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp2Z
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2Р
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2"
LSTM/lstm/whileLSTM/lstm/while:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Љ
serving_default®
?
input_14
serving_default_input_1:0€€€€€€€€€dI
tf.math.l2_normalize1
StatefulPartitionedCall:0€€€€€€€€€Аtensorflow/serving/predict: Х
в
layer-0
layer_with_weights-0
layer-1
layer-2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
=__call__
*>&call_and_return_all_conditional_losses
?_default_save_signature"÷
_tf_keras_networkЇ{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Model", "config": {"layer was saved without config": true}, "name": "LSTM", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.l2_normalize", "trainable": true, "dtype": "float32", "function": "math.l2_normalize"}, "name": "tf.math.l2_normalize", "inbound_nodes": [["LSTM", 0, 0, {"axis": 1, "epsilon": 1e-10}]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["tf.math.l2_normalize", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "loss", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adagrad", "config": {"name": "Adagrad", "learning_rate": 0.10000000149011612, "decay": 0.0, "initial_accumulator_value": 0.1, "epsilon": 1e-07}}}}
у"р
_tf_keras_input_layer–{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
Й

lstm
regularization_losses
trainable_variables
	variables
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"р
_tf_keras_model÷{"class_name": "Model", "name": "LSTM", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Model"}}
т
	keras_api"а
_tf_keras_layer∆{"class_name": "TFOpLambda", "name": "tf.math.l2_normalize", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.l2_normalize", "trainable": true, "dtype": "float32", "function": "math.l2_normalize"}}
t
iter
	decay
learning_rateaccumulator:accumulator;accumulator<"
	optimizer
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 
regularization_losses

layers
non_trainable_variables
layer_metrics
trainable_variables
layer_regularization_losses
metrics
	variables
=__call__
?_default_save_signature
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
,
Bserving_default"
signature_map
Ь
cell

state_spec
regularization_losses
trainable_variables
	variables
 	keras_api
C__call__
*D&call_and_return_all_conditional_losses"у

_tf_keras_rnn_layer’
{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 2]}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
≠
regularization_losses

!layers
"non_trainable_variables
#layer_metrics
trainable_variables
$layer_regularization_losses
%metrics
	variables
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
:	 (2Adagrad/iter
: (2Adagrad/decay
: (2Adagrad/learning_rate
-:+	А2LSTM/lstm/lstm_cell/kernel
8:6
АА2$LSTM/lstm/lstm_cell/recurrent_kernel
':%А2LSTM/lstm/lstm_cell/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
&0"
trackable_list_wrapper
И	

kernel
recurrent_kernel
bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
E__call__
*F&call_and_return_all_conditional_losses"Ќ
_tf_keras_layer≥{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
є

+states
regularization_losses

,layers
-non_trainable_variables
.layer_metrics
trainable_variables
/layer_regularization_losses
0metrics
	variables
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ї
	1total
	2count
3	variables
4	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
.
G0
H1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
≠
'regularization_losses

5layers
6non_trainable_variables
7layer_metrics
(trainable_variables
8layer_regularization_losses
9metrics
)	variables
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
10
21"
trackable_list_wrapper
-
3	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?:=	А2.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulator
J:H
АА28Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator
9:7А2,Adagrad/LSTM/lstm/lstm_cell/bias/accumulator
о2л
(__inference_model_layer_call_fn_11856942
(__inference_model_layer_call_fn_11856953
(__inference_model_layer_call_fn_11856517
(__inference_model_layer_call_fn_11856558ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
C__inference_model_layer_call_and_return_conditional_losses_11856475
C__inference_model_layer_call_and_return_conditional_losses_11856759
C__inference_model_layer_call_and_return_conditional_losses_11856931
C__inference_model_layer_call_and_return_conditional_losses_11856445ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
е2в
#__inference__wrapped_model_11855100Ї
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ **Ґ'
%К"
input_1€€€€€€€€€d
Ё2Џ
'__inference_LSTM_layer_call_fn_11857317
'__inference_LSTM_layer_call_fn_11857306
'__inference_LSTM_layer_call_fn_11856227
'__inference_LSTM_layer_call_fn_11856216≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
…2∆
B__inference_LSTM_layer_call_and_return_conditional_losses_11857295
B__inference_LSTM_layer_call_and_return_conditional_losses_11856181
B__inference_LSTM_layer_call_and_return_conditional_losses_11857130
B__inference_LSTM_layer_call_and_return_conditional_losses_11856158≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЌB 
&__inference_signature_wrapper_11856587input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€2ь
'__inference_lstm_layer_call_fn_11858022
'__inference_lstm_layer_call_fn_11857670
'__inference_lstm_layer_call_fn_11857681
'__inference_lstm_layer_call_fn_11858033’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
л2и
B__inference_lstm_layer_call_and_return_conditional_losses_11858011
B__inference_lstm_layer_call_and_return_conditional_losses_11857659
B__inference_lstm_layer_call_and_return_conditional_losses_11857846
B__inference_lstm_layer_call_and_return_conditional_losses_11857494’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
†2Э
,__inference_lstm_cell_layer_call_fn_11858152
,__inference_lstm_cell_layer_call_fn_11858169Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
G__inference_lstm_cell_layer_call_and_return_conditional_losses_11858090
G__inference_lstm_cell_layer_call_and_return_conditional_losses_11858135Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
µ2≤
__inference_loss_fn_0_11858180П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
µ2≤
__inference_loss_fn_1_11858191П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ ≠
B__inference_LSTM_layer_call_and_return_conditional_losses_11856158g8Ґ5
.Ґ+
%К"
input_1€€€€€€€€€d
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ≠
B__inference_LSTM_layer_call_and_return_conditional_losses_11856181g8Ґ5
.Ґ+
%К"
input_1€€€€€€€€€d
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ђ
B__inference_LSTM_layer_call_and_return_conditional_losses_11857130f7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€d
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ђ
B__inference_LSTM_layer_call_and_return_conditional_losses_11857295f7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€d
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ Е
'__inference_LSTM_layer_call_fn_11856216Z8Ґ5
.Ґ+
%К"
input_1€€€€€€€€€d
p
™ "К€€€€€€€€€АЕ
'__inference_LSTM_layer_call_fn_11856227Z8Ґ5
.Ґ+
%К"
input_1€€€€€€€€€d
p 
™ "К€€€€€€€€€АД
'__inference_LSTM_layer_call_fn_11857306Y7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€d
p
™ "К€€€€€€€€€АД
'__inference_LSTM_layer_call_fn_11857317Y7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€d
p 
™ "К€€€€€€€€€А±
#__inference__wrapped_model_11855100Й4Ґ1
*Ґ'
%К"
input_1€€€€€€€€€d
™ "L™I
G
tf.math.l2_normalize/К,
tf.math.l2_normalize€€€€€€€€€А=
__inference_loss_fn_0_11858180Ґ

Ґ 
™ "К =
__inference_loss_fn_1_11858191Ґ

Ґ 
™ "К ќ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_11858090ВВҐ
xҐu
 К
inputs€€€€€€€€€
MҐJ
#К 
states/0€€€€€€€€€А
#К 
states/1€€€€€€€€€А
p
™ "vҐs
lҐi
К
0/0€€€€€€€€€А
GЪD
 К
0/1/0€€€€€€€€€А
 К
0/1/1€€€€€€€€€А
Ъ ќ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_11858135ВВҐ
xҐu
 К
inputs€€€€€€€€€
MҐJ
#К 
states/0€€€€€€€€€А
#К 
states/1€€€€€€€€€А
p 
™ "vҐs
lҐi
К
0/0€€€€€€€€€А
GЪD
 К
0/1/0€€€€€€€€€А
 К
0/1/1€€€€€€€€€А
Ъ £
,__inference_lstm_cell_layer_call_fn_11858152тВҐ
xҐu
 К
inputs€€€€€€€€€
MҐJ
#К 
states/0€€€€€€€€€А
#К 
states/1€€€€€€€€€А
p
™ "fҐc
К
0€€€€€€€€€А
CЪ@
К
1/0€€€€€€€€€А
К
1/1€€€€€€€€€А£
,__inference_lstm_cell_layer_call_fn_11858169тВҐ
xҐu
 К
inputs€€€€€€€€€
MҐJ
#К 
states/0€€€€€€€€€А
#К 
states/1€€€€€€€€€А
p 
™ "fҐc
К
0€€€€€€€€€А
CЪ@
К
1/0€€€€€€€€€А
К
1/1€€€€€€€€€Аі
B__inference_lstm_layer_call_and_return_conditional_losses_11857494n?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€d

 
p

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ і
B__inference_lstm_layer_call_and_return_conditional_losses_11857659n?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€d

 
p 

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ƒ
B__inference_lstm_layer_call_and_return_conditional_losses_11857846~OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ƒ
B__inference_lstm_layer_call_and_return_conditional_losses_11858011~OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ М
'__inference_lstm_layer_call_fn_11857670a?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€d

 
p

 
™ "К€€€€€€€€€АМ
'__inference_lstm_layer_call_fn_11857681a?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€d

 
p 

 
™ "К€€€€€€€€€АЬ
'__inference_lstm_layer_call_fn_11858022qOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€АЬ
'__inference_lstm_layer_call_fn_11858033qOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€А≤
C__inference_model_layer_call_and_return_conditional_losses_11856445k<Ґ9
2Ґ/
%К"
input_1€€€€€€€€€d
p

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ≤
C__inference_model_layer_call_and_return_conditional_losses_11856475k<Ґ9
2Ґ/
%К"
input_1€€€€€€€€€d
p 

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ±
C__inference_model_layer_call_and_return_conditional_losses_11856759j;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€d
p

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ±
C__inference_model_layer_call_and_return_conditional_losses_11856931j;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€d
p 

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ К
(__inference_model_layer_call_fn_11856517^<Ґ9
2Ґ/
%К"
input_1€€€€€€€€€d
p

 
™ "К€€€€€€€€€АК
(__inference_model_layer_call_fn_11856558^<Ґ9
2Ґ/
%К"
input_1€€€€€€€€€d
p 

 
™ "К€€€€€€€€€АЙ
(__inference_model_layer_call_fn_11856942];Ґ8
1Ґ.
$К!
inputs€€€€€€€€€d
p

 
™ "К€€€€€€€€€АЙ
(__inference_model_layer_call_fn_11856953];Ґ8
1Ґ.
$К!
inputs€€€€€€€€€d
p 

 
™ "К€€€€€€€€€Ањ
&__inference_signature_wrapper_11856587Ф?Ґ<
Ґ 
5™2
0
input_1%К"
input_1€€€€€€€€€d"L™I
G
tf.math.l2_normalize/К,
tf.math.l2_normalize€€€€€€€€€А