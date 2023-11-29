        cache_mask_list = []
        xv_extended_list = []
        
        if start_pos > 0:
            cache_mask_list.append(torch.ones((self.cache_v.shape[0],start_pos,self.cache_v.shape[2],self.cache_v.shape[3])))
            xv_extended_list.append(torch.zeros((self.cache_v.shape[0],start_pos,self.cache_v.shape[2],self.cache_v.shape[3])))

        cache_mask_list.append(torch.zeros(xv.shape))
        xv_extended_list.append(xv)

        if (start_pos + seqlen) < self.cache_v.shape[1]:
            cache_mask_list.append(torch.ones((self.cache_v.shape[0],self.cache_v.shape[1] - (start_pos + seqlen),self.cache_v.shape[2],self.cache_v.shape[3])))
            xv_extended_list.append(torch.zeros((self.cache_v.shape[0],self.cache_v.shape[1] - (start_pos + seqlen),self.cache_v.shape[2],self.cache_v.shape[3])))

        cache_mask = torch.cat(cache_mask_list,dim=1)
        xv_extended = torch.cat(xv_extended_list,dim=1)
        
        self.cache_v = self.cache_v*cache_mask + xv_extended