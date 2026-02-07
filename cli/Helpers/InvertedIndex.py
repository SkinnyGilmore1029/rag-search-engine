class InvertedIndex:
    def __init__(self, index:dict[str, set[int]], docmap:dict[int, object]):
        self.index = index
        self.docmap = docmap
        
    def __add_documents(self, doc_id:int, text:str) -> None:
        pass
    
    def get_documents(self, term:any) -> list[int]:
        pass
    
    def build(self) -> None:
        pass
    
    def save(self) -> None:
        pass
    
    