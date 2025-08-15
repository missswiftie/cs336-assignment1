from typing import Iterable,Iterator,List,Dict,Tuple
import os
import regex as re
import heapq
from array import array

PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(text:str)->List[bytes]:
    split_token_list=re.findall(PAT,text)
    byted_tokens=[s.encode('utf-8') for s in split_token_list]
    return byted_tokens

PAT_RE = re.compile(PAT)

def iterable_pretokenize(text: str) -> Iterator[bytes]:
    for m in PAT_RE.finditer(text):
        yield m.group(0).encode("utf-8")

class BytePair:
    def __init__(self,count,token_id1,token_id2,itos):
        self.count=count
        self.token_id1=token_id1
        self.token_id2=token_id2
        self.byte1=itos[token_id1]
        self.byte2=itos[token_id2]
    def __lt__(self,other):
        if self.count!=other.count:
            return self.count>other.count
        if self.byte1!=other.byte1:
            return self.byte1>other.byte1
        return self.byte2>other.byte2
    def __eq__(self,other):
        return self.count==other.count and self.byte1==other.byte1 and self.byte2==other.byte2
    def get_pair(self):
        return (self.token_id1,self.token_id2)

class BPETokenizer:
    def __init__(self,vocab_size:int,special_tokens:list[str]|None=None):
        self.vocab_size=vocab_size#int
        self.special_tokens=special_tokens if special_tokens is not None else []#list[str]
        self.special_tokens_bytes=[s.encode('utf-8') for s in self.special_tokens]#list[bytes]
        self.special_tokens_num=len(self.special_tokens_bytes)
        self.merges:List[Tuple[bytes,bytes]]=[]
        self.stoi:Dict[bytes,int]={}
        self.itos:Dict[int,bytes]={}
        self.merges_rank:Dict[Tuple[bytes,bytes],int]={}

        for i in range(256):
            self.stoi[bytes([i])]=i
            self.itos[i]=bytes([i])

        for i,special_token in enumerate(self.special_tokens_bytes):
            self.stoi[special_token]=i+256
            self.itos[i+256]=special_token

        self.vocab=self.itos.copy()
        self.pair_to_new={}

    def get_state(self,token_groups:List[List[int]]):
        counts={}
        for token in token_groups:
            for pair in zip(token,token[1:]):
                counts[pair]=counts.get(pair,0)+1
        return counts
    
    def merge_tokens(self,tokens_group:List[List[int]],max_pair:Tuple[int,int],new_index:int):
        result=[]
        for tokens in tokens_group:
            token_result=[]
            i=0
            while i<len(tokens):
                if i<len(tokens)-1 and (tokens[i],tokens[i+1])==max_pair:
                    token_result.append(new_index)
                    i+=2
                else:
                    token_result.append(tokens)
                    i+=1
            result.append(token_result)
        return result
    
    def train_bpe(self,input_path:str|os.PathLike):
        with open(input_path,'r',encoding='utf-8') as f:
            text=f.read()
        if self.special_tokens:
            split_pattern=f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
            parts=re.split(split_pattern,text)
        else:
            parts=[text]
        token_groups=[]#list[list[int]]
        for part in parts:
            if part in self.special_tokens or not part:
                continue
            word_in_bytes=pretokenize(part)
            for word in word_in_bytes:
                token_groups.append([self.stoi[bytes([b])] for b in word])
        
        idx=0
        counts={}
        token={}
        pre={}
        nxt={}
        pos={}
        for i,tokens in enumerate(token_groups):
            if not tokens or len(tokens)<=1:
                continue
            size=len(tokens)
            for j,token_id in enumerate(tokens):
                idx+=1
                token[idx]=token_id
                nxt[idx]=None if j==size-1 else idx+1
                pre[idx]=None if j==0 else idx-1
                if j==size-1:
                    continue
                token_pair=(token_id,tokens[j+1])
                counts[token_pair]=counts.get(token_pair,0)+1
                if pos.get(token_pair) is None:
                    pos[token_pair]=set()
                pos[token_pair].add(idx)
        
        heap = [BytePair(count, a, b, self.itos) for (a, b), count in counts.items()]
        heapq.heapify(heap)
        
        num_merges=self.vocab_size-len(self.stoi)


        def merge(pair:Tuple[int,int],delta:int,pos_idx:int|None=None):
            if pair is None or None in pair:
                return 
            counts[pair]=counts.get(pair,0)+delta
            count_=counts[pair]
            if count_<=0:
                counts.pop(pair,None)
                pos.pop(pair,None)
                return 
            if pos_idx is not None:
                pos_set=pos.setdefault(pair,set())
                if delta>0:
                    pos_set.add(pos_idx)
                else:
                    pos_set.discard(pos_idx)
            item=BytePair(count_,pair[0],pair[1],self.itos)
            heapq.heappush(heap,item)


        while num_merges>0 and heap:
            if not counts:
                break
            num_merges-=1
            while heap:
                item=heapq.heappop(heap)
                index1,index2=item.get_pair()
                if (index1,index2) not in counts or counts[(index1,index2)]!=item.count:
                    continue
                self.merges.append((self.itos[index1],self.itos[index2]))
                byte1,byte2=self.itos[index1],self.itos[index2]
                max_pair=byte1+byte2
                new_index=len(self.stoi) if self.stoi.get(max_pair) is None else self.stoi[max_pair]
                self.pair_to_new[index1,index2]=new_index
                self.stoi[max_pair]=new_index
                self.itos[new_index]=max_pair
                pos_lst=list(pos.get((index1,index2),set()))
                for pos_idx in pos_lst:
                    pre_idx=pre[pos_idx]
                    nxt_idx=nxt[pos_idx]
                    nnxt_idx=nxt[nxt_idx] if nxt_idx is not None else None
                    if nxt_idx is None or token[pos_idx]!=index1 or token[nxt_idx]!=index2:
                        continue
                    if pre_idx is not None:
                        nxt[pre_idx]=pos_idx
                        merge((token[pre_idx],token[pos_idx]),-1,pre_idx)
                        merge((token[pre_idx],new_index),1,pre_idx)
                    if nnxt_idx is not None:
                        pre[nnxt_idx]=pos_idx
                        merge((token[nxt_idx],token[nnxt_idx]),-1,nxt_idx)
                        merge((new_index,token[nnxt_idx]),1,pos_idx)

                    pre[pos_idx]=pre_idx
                    nxt[pos_idx]=nnxt_idx
                    token[pos_idx]=new_index
                    token[nxt_idx]=None
                    pre[nxt_idx]=None
                    nxt[nxt_idx]=None
                
                counts.pop((index1,index2),None)
                pos.pop((index1,index2),None)
                break

        self.vocab=self.itos.copy()
        self.merges_rank={pair:k for k,pair in enumerate(self.merges)}


    def encode_single_text(self,text:str)->List[int]:
        if text is None:
            return []
        result=array('H')
        for word in iterable_pretokenize(text):
            token_int=array('H',(self.stoi[bytes([b])] for b in word))
            while True:   
                best_rank=1000000000
                best_pos=-1
                for i in range(len(token_int)-1):
                    current_rank=self.merges_rank.get((self.itos[token_int[i]],self.itos[token_int[i+1]]),1000000000)
                    if best_rank>current_rank:
                        best_rank,best_pos=current_rank,i
                if best_pos<0:
                    break
                # if len(text)==1:
                #     print(token_int,best_pos)
                #     print(token_int[best_pos])
                token_int[best_pos:best_pos+2]=array('H',[self.pair_to_new[(token_int[best_pos],token_int[best_pos+1])]])
            result.extend(token_int)
        return result.tolist()

    def encode(self,text:str)->List[int]:
        if self.special_tokens:
            sorted_special_tokens=sorted(self.special_tokens,key=lambda x:-len(x))
            split_pattern=f"({'|'.join(re.escape(s) for s in sorted_special_tokens)})"
            parts=re.split(split_pattern,text)
        else:
            return self.encode_single_text(text)
        token_groups=[]#list[list[int]]
        for part in parts:
            if part in self.special_tokens:
                token_groups.append(self.stoi[part.encode('utf-8')])
            elif part:
                token_groups.extend(self.encode_single_text(part))
        return token_groups
    
    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
        for line in iterable:
            yield from self.encode(line)
    
    def decode(self,token_int:List[int])->str:
        byte_text=b"".join(self.vocab[idn] for idn in token_int)
        return byte_text.decode('utf-8',errors="replace")
    
    @classmethod
    def from_files(cls,vocab:Dict[int,bytes],merges:List[Tuple[bytes,bytes]],special_tokens:List[str]):
        instance=cls(len(vocab),special_tokens)
        instance.merges=merges
        instance.vocab=vocab
        instance.itos=vocab
        instance.stoi={v:k for k,v in vocab.items()}
        instance.merges_rank={pair:i for i,pair in enumerate(merges)}
        instance.pair_to_new={(instance.stoi[p1],instance.stoi[p2]):instance.stoi[p1+p2] for p1,p2 in merges}
        return instance