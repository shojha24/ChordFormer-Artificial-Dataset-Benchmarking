import torch.nn as nn
import torch.nn.functional as F
from mir.nn.train import NetworkBehavior,NetworkInterface
from mir.nn.data_storage import FramedRAMDataStorage,FramedH5DataStorage
from mir.nn.data_decorator import CQTPitchShifter,AbstractPitchShifter,NoPitchShifter
from mir.nn.data_provider import FramedDataProvider
import torch
import numpy as np
from complex_chord import Chord,ChordTypeLimit,shift_complex_chord_array_list,complex_chord_chop,enum_to_dict,\
    TriadTypes,SeventhTypes,NinthTypes,EleventhTypes,ThirteenthTypes,complex_chord_chop_list
from train_eval_test_split import get_train_set_ids,get_test_set_ids,get_val_set_ids

def set_specific_gpu(gpu_index=5):
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        if gpu_index < total_gpus:
            torch.cuda.set_device(gpu_index)
            print(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
        else:
            raise ValueError(f"GPU index {gpu_index} is out of range. Only {total_gpus} GPUs available.")
    else:
        raise RuntimeError("No GPU is available.")

set_specific_gpu()  # This will set the third GPU

SHIFT_LOW=-5
SHIFT_HIGH=6
SHIFT_STEP=3
SPEC_DIM=252
LSTM_TRAIN_LENGTH=1000

chord_limit=ChordTypeLimit(
    triad_limit=6,
    seventh_limit=3,
    ninth_limit=3,
    eleventh_limit=2,
    thirteenth_limit=2
)


class ReweightedLoss(nn.Module):

    def __init__(self,counter,power=1.0,max_clip=1.0,gpu=False,triad_only=False):
        super(ReweightedLoss, self).__init__()
        self.weight=[None]*6
        for i in range(6):
            if(i==0 or i==1):
                self.weight[i]=torch.tensor([counter[i][(j+11)//12] for j in range(len(counter[i])*12-11)],dtype=torch.float32)
            else:
                self.weight[i]=torch.tensor(counter[i],dtype=torch.float32)
            self.weight[i]=torch.pow(self.weight[i].max()/self.weight[i],power)
            self.weight[i][self.weight[i]>max_clip]=max_clip
            if(gpu==True):
                self.weight[i]=self.weight[i].cuda()
        self.triad_only=triad_only


    def forward(self, output, tag):
        def conditional_classifier_loss(a,b,weight=None):
            if((b<0).all()):
                return torch.tensor(0,device=b.device)
            loss=F.cross_entropy(a[b>=0],b[b>=0],weight=weight[:a.shape[1]])
            #loss_term=self.loss_calc(a[b>=0],b[b>=0])
            return loss
        if(self.triad_only):
            result=conditional_classifier_loss(output[0],tag[:,0],weight=self.weight[0])
        else:
            result=conditional_classifier_loss(output[0],tag[:,0],weight=self.weight[0])+\
                conditional_classifier_loss(output[1],tag[:,1]+1,weight=self.weight[1])+\
                conditional_classifier_loss(output[2],tag[:,2],weight=self.weight[2])+\
                conditional_classifier_loss(output[3],tag[:,3],weight=self.weight[3])+\
                conditional_classifier_loss(output[4],tag[:,4],weight=self.weight[4])+\
                conditional_classifier_loss(output[5],tag[:,5],weight=self.weight[5])
        return result





from typing import Optional, Tuple


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)


class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        encoder_padding_mask = _lengths_to_padding_mask(lengths)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1), lengths

class ChordConformer(torch.nn.Module):
    """
    Conformer architecture adapted for chord recognition with fixed-length audio inputs.
    """

    def __init__(self, input_dim, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size, output_dim):
        super().__init__()

        self.conformer_layers = nn.ModuleList([
            ConformerLayer(
                input_dim=input_dim,
                ffn_dim=ffn_dim,
                num_attention_heads=num_heads,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=0.1,
                use_group_norm=True,
                convolution_first=True
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(input_dim, output_dim)


    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Shape `(B, T, input_dim)` where B is batch size, T is sequence length, and input_dim is the feature dimension.
        """
        x = input.transpose(0, 1)  # Conformers expect input in (T, B, D) format
        for layer in self.conformer_layers:
            x = layer(x, None)  # No padding mask needed as all inputs are the same length
        x = x.transpose(0, 1)  # Convert back to (B, T, D)
        x = self.output_layer(x)
  

        return x


class ChordNet(NetworkBehavior):

    def __init__(self,cross_subpart_counter,triad_only=False):
        super(ChordNet, self).__init__()
        self.triad_only=triad_only

        self.chordfor = ChordConformer(input_dim=256, num_heads=16, ffn_dim=1024, num_layers=4, depthwise_conv_kernel_size=31, output_dim=100)
  

        self.hidden_dim1=192
        
        self.output_dim1=chord_limit.triad_limit*12+2+12
        self.output_dim2=chord_limit.seventh_limit+chord_limit.ninth_limit+chord_limit.eleventh_limit+chord_limit.thirteenth_limit+4
        self.final_fc1=nn.Linear(self.hidden_dim1,self.output_dim1+self.output_dim2)

        #self.loss_calc=FocalLoss(gamma=2.0)
        if(cross_subpart_counter is not None):
            self.loss_reweight=ReweightedLoss(cross_subpart_counter,power=1.0,max_clip=1.0,gpu=self.use_gpu,triad_only=triad_only)
    

    def forward(self, x):
        batch_size=x.shape[0]
        seq_length=x.shape[1]
        padding = (0, 4, 0, 0, 0, 0)  # Adds 4 units of padding only to the last dimension
        x = torch.nn.functional.pad(x, padding, "constant", 0)
        x = x.to(torch.float32)
        
        x=self.chordfor(x)
        x1=x.reshape((batch_size*seq_length,self.output_dim1+self.output_dim2))

        bass_del=chord_limit.bass_slice_begin+12+1
        seventh_del=bass_del+chord_limit.seventh_limit+1
        ninth_del=seventh_del+chord_limit.ninth_limit+1
        eleventh_del=ninth_del+chord_limit.eleventh_limit+1
        thirteenth_del=eleventh_del+chord_limit.thirteenth_limit+1
        return x1[:,:chord_limit.bass_slice_begin],\
            x1[:,chord_limit.bass_slice_begin:bass_del],\
            x1[:,bass_del:seventh_del],\
            x1[:,seventh_del:ninth_del],\
            x1[:,ninth_del:eleventh_del],\
            x1[:,eleventh_del:thirteenth_del]

    def loss(self, x, y):
        output=self.feed(x)
        tag=y.view((-1,6))
        return self.loss_reweight(output,tag)

    def inference(self, x):
        seq_length=x.shape[0]
        output=self.feed(x[:,SHIFT_HIGH*SHIFT_STEP:SHIFT_HIGH*SHIFT_STEP+SPEC_DIM].view((1,seq_length,SPEC_DIM)))
        result_triad=F.softmax(output[0],dim=1).cpu().numpy()
        result_bass=F.softmax(output[1],dim=1).cpu().numpy()
        result_7=F.softmax(output[2],dim=1).cpu().numpy()
        result_9=F.softmax(output[3],dim=1).cpu().numpy()
        result_11=F.softmax(output[4],dim=1).cpu().numpy()
        result_13=F.softmax(output[5],dim=1).cpu().numpy()
        return result_triad,result_bass,result_7,result_9,result_11,result_13

class ChordNetCNN(NetworkBehavior):

    def __init__(self,cross_subpart_counter):
        super(ChordNetCNN, self).__init__()
        self.audio_feature_block=CNNFeatureExtractor()

        self.hidden_dim1=192
        self.output_dim1=chord_limit.triad_limit*12+2+12
        self.output_dim2=chord_limit.seventh_limit+chord_limit.ninth_limit+chord_limit.eleventh_limit+chord_limit.thirteenth_limit+4
        self.final_fc1=nn.Linear(self.audio_feature_block.output_size,self.output_dim1+self.output_dim2)

        #self.loss_calc=FocalLoss(gamma=2.0)
        if(cross_subpart_counter is not None):
            self.loss_reweight=ReweightedLoss(cross_subpart_counter,power=1.0,max_clip=1.0,gpu=self.use_gpu,triad_only=triad_only)
    def init_hidden(self,batch_size,hidden_dim):
        c_0=torch.zeros(2,batch_size,hidden_dim//2)
        h_0=torch.zeros(2,batch_size,hidden_dim//2)
        if(self.use_gpu):
            c_0=c_0.cuda()
            h_0=h_0.cuda()
        return (c_0,h_0)

    def forward(self, x):
        batch_size=x.shape[0]
        seq_length=x.shape[1]
        x=self.audio_feature_block(x)
        x1=self.final_fc1(x).reshape((batch_size*seq_length,self.output_dim1+self.output_dim2))

        bass_del=chord_limit.bass_slice_begin+12+1
        seventh_del=bass_del+chord_limit.seventh_limit+1
        ninth_del=seventh_del+chord_limit.ninth_limit+1
        eleventh_del=ninth_del+chord_limit.eleventh_limit+1
        thirteenth_del=eleventh_del+chord_limit.thirteenth_limit+1
        return x1[:,:chord_limit.bass_slice_begin],\
            x1[:,chord_limit.bass_slice_begin:bass_del],\
            x1[:,bass_del:seventh_del],\
            x1[:,seventh_del:ninth_del],\
            x1[:,ninth_del:eleventh_del],\
            x1[:,eleventh_del:thirteenth_del]

    def loss(self, x, y):
        output=self.feed(x)
        tag=y.view((-1,6))
        return self.loss_reweight(output,tag)

    def inference(self, x):
        seq_length=x.shape[0]
        output=self.feed(x[:,SHIFT_HIGH*SHIFT_STEP:SHIFT_HIGH*SHIFT_STEP+SPEC_DIM].view((1,seq_length,SPEC_DIM)))
        result_triad=F.softmax(output[0],dim=1).cpu().numpy()
        result_bass=F.softmax(output[1],dim=1).cpu().numpy()
        result_7=F.softmax(output[2],dim=1).cpu().numpy()
        result_9=F.softmax(output[3],dim=1).cpu().numpy()
        result_11=F.softmax(output[4],dim=1).cpu().numpy()
        result_13=F.softmax(output[5],dim=1).cpu().numpy()
        return result_triad,result_bass,result_7,result_9,result_11,result_13

class FocalLoss(nn.Module):

    def __init__(self,gamma=0.0):
        super(FocalLoss, self).__init__()
        self.gamma=gamma

    def forward(self, input, target):
        logpt=F.log_softmax(input,dim=1)
        logpt=logpt.gather(1,target[:,None]).view((-1))
        pt=torch.tensor(logpt.data.exp())
        loss=-1*(1-pt)**self.gamma*logpt
        return loss.mean()

class ComplexChordShifter(AbstractPitchShifter):

    def pitch_shift(self,data,shift):
        return shift_complex_chord_array_list(complex_chord_chop_list(data,chord_limit),shift)

if __name__ == '__main__':
    TOTAL_SLICE_COUNT=5
    import sys,pickle
    slice_id=int(sys.argv[1])
    if(slice_id>=5 or slice_id<-1):
        raise Exception('Invalid input')
    storage_x=FramedH5DataStorage('jams_cqt')
    storage_y=FramedH5DataStorage('jams_xchord')
    storage_x.load_meta()
    song_count=storage_x.total_song_count
    if(0<=slice_id and slice_id<=5):
        print('Train on slice %d'%slice_id)
        f=open('data/cross_subpart_weight%d.pkl'%slice_id,'rb')
        cross_subpart_counter=pickle.load(f)
        f.close()
        train_indices=get_train_set_ids(slice_id)
        val_indices=get_val_set_ids(slice_id)
    else:
        train_indices=np.arange(song_count)
        val_indices=np.arange(0,1) # fake validation here
        # todo: weight calculation for full dataset
        f=open('data/cross_subpart_weight%d.pkl'%0,'rb')
        cross_subpart_counter=pickle.load(f)
        f.close()
    train_provider=FramedDataProvider(train_sample_length=LSTM_TRAIN_LENGTH,shift_low=SHIFT_LOW,shift_high=SHIFT_HIGH,num_workers=4,average_samples_per_song=1)
    train_provider.link(storage_x,CQTPitchShifter(SPEC_DIM,SHIFT_LOW,SHIFT_HIGH),subrange=train_indices)
    train_provider.link(storage_y,ComplexChordShifter(),subrange=train_indices)

    val_provider=FramedDataProvider(train_sample_length=-1,shift_low=0,shift_high=0,num_workers=4,average_samples_per_song=1,need_shuffle=False)
    val_provider.link(storage_x,CQTPitchShifter(SPEC_DIM,SHIFT_LOW,SHIFT_HIGH),subrange=val_indices)
    val_provider.link(storage_y,ComplexChordShifter(),subrange=val_indices)



    trainer=NetworkInterface(ChordNet(cross_subpart_counter,triad_only=False),
                             'chordformer_head16(1.0,1.0)_s%d'%slice_id,load_checkpoint=True)
    print(trainer)
    if(slice_id==-1):
    	trainer.train_supervised(train_provider,val_provider,batch_size=24,
                             learning_rates_dict={1e-3:35,1e-4:25,1e-5:15,1e-6:10},round_per_print=10,round_per_save=500,
                             round_per_val=-1,early_end_epochs=100,val_batch_size=1)
    else:
    	trainer.train_supervised(train_provider,val_provider,batch_size=24,
                                learning_rates_dict={1e-3:60,1e-4:30,1e-5:30,1e-6:10},round_per_print=10,round_per_save=500,
                                 round_per_val=-1,early_end_epochs=5,val_batch_size=1)
    