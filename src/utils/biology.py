"""
Authors: Nicolas Raymond

Description: Contains classes that make abstraction of certain biology concept.
"""
from Bio.Seq import Seq

class Chromosome:
    """
    Class that makes abstraction of a chromosome.
    """
    def __init__(self,
                 chrom_id: str,
                 seq: Seq) -> None:
        """
        Saves the id and the nucleotides sequence.
        
        Args:
            id (str): id of the chromosome (e.g., 'Chr1').
            seq (Seq): sequence of nucleotides.
        """
        self.id: str = chrom_id
        self.seq: Seq = seq

    def __len__(self) -> int:
        """
        Returns the length of the nucleotides sequence.
        
        Returns:
            int: length of the nucleotides sequence.
        """
        return len(self.seq)

    def to_list(self) -> list[str, Seq]:
        """
        Return all the attributes in a list.
        
        Returns:
            list[str, Seq]: ['id', 'seq]
        """
        return [self.id, self.seq]


class Gene:
    """
    Class that makes abstraction of a gene.
    """
    def __init__(self,
                 gene_id: str,
                 chromosome_id: str,
                 start: int,
                 end: int,
                 sense_strand: bool,
                 seq: str) -> None:
        """
        Sets all public attributes.
        
        Args:
            id (str): sequence of characters identifying the gene.
            chromosome_id (str): id of the chromosome on which the gene is located.
            start (int): idx indicating where the gene starts on the chromosome.
            end (int): idx indicating where the gene ends on the chromosome.
            sense_strand (str): if True, the strand orientation is 5' -> 3'.
                                The opposite otherwise
            seq (Seq): sequence of nucleotides
        """
        self.id: str = gene_id
        self.chromosome_id: str = chromosome_id
        self.start: int = start
        self.end: int = end
        self.sense_strand: bool = sense_strand
        self.seq: Seq = seq

    def __len__(self) -> int:
        """
        Returns the length of the nucleotides sequence.
        
        Returns:
            int: length of the nucleotides sequence.
        """
        return len(self.seq)

    def to_list(self) -> list[str, str, int, int, Seq, bool]:
        """
        Return all the attributes in a list.
        
        Returns:
            list[str, str, int, int, str]: ['chromosome id', 'id',
                                            'start', 'end',
                                            'sense strand', 'seq']
        """
        return [self.id, self.chromosome_id,
                self.start, self.end,
                self.sense_strand, self.seq]
