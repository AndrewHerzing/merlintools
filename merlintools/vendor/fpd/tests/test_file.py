
import unittest
import os
import zipfile
import tarfile
import tempfile
import shutil
import matplotlib.pylab as plt
plt.ion()
import numpy as np
import h5py

from fpd.fpd_file import MerlinBinary, DataBrowser
import fpd.fpd_file as fpdf


class TestFile(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tmp_d = tempfile.mkdtemp(prefix='fpd-')
        self.bp = os.path.dirname(__file__)
        
        # multiple images per file
        z_fn = os.path.join(self.bp, 'file_datasets', 'medipix_test_binary_data.zip')
        with zipfile.ZipFile(z_fn, 'r') as z:
            z.extractall(path=self.tmp_d)
            z_fns = z.namelist()
        self.z_dm_fns = [x for x in z_fns if x.split('.')[-1].lower() in ['dm3', 'dm4']]
        self.z_bn_fns = [x for x in z_fns if x.split('.')[-1].lower() in ['bin', 'mib']]
        self.z_hd_fns = [x for x in z_fns if x.split('.')[-1].lower() in ['hdr']]
        self.z_dm_fns.sort()
        self.z_bn_fns.sort()
        self.z_hd_fns.sort()
        
        # multiple single-image files
        self.tmp_d_multi_ims = tempfile.mkdtemp(dir=self.tmp_d, prefix='multiple_single_images-')
        z_fn = os.path.join(self.bp, 'file_datasets', 'zero_values_6bit_multiple_files_16x16.zip')
        with zipfile.ZipFile(z_fn, 'r') as z:
            z.extractall(path=self.tmp_d_multi_ims)
            z_multi_ims_fns = z.namelist()
        self.z_mi_dm_fns = [x for x in z_multi_ims_fns if x.split('.')[-1].lower() in ['dm3', 'dm4']]
        self.z_mi_bn_fns = [x for x in z_multi_ims_fns if x.split('.')[-1].lower() in ['bin', 'mib']]
        self.z_mi_hd_fns = [x for x in z_multi_ims_fns if x.split('.')[-1].lower() in ['hdr']]
        
        # raw mode
        self.tmp_d_raw = tempfile.mkdtemp(dir=self.tmp_d, prefix='raw-')
        z_fn = os.path.join(self.bp, 'file_datasets', 'medipix_test_binary_data_raw.zip')
        with zipfile.ZipFile(z_fn, 'r') as z:
            z.extractall(path=self.tmp_d_raw)
            z_fns = z.namelist()
        self.z_dm_fns_raw = [x for x in z_fns if x.split('.')[-1].lower() in ['dm3', 'dm4']]
        self.z_bn_fns_raw = [x for x in z_fns if x.split('.')[-1].lower() in ['bin', 'mib']]
        self.z_hd_fns_raw = [x for x in z_fns if x.split('.')[-1].lower() in ['hdr']]
        self.z_dm_fns_raw.sort()
        self.z_bn_fns_raw.sort()
        self.z_hd_fns_raw.sort()
    
        # raw and non-raw image data (non-blank)
        self.tmp_d_images = tempfile.mkdtemp(dir=self.tmp_d, prefix='image-')
        z_fn = os.path.join(self.bp, 'file_datasets', 'medipix_test_binary_data_images.zip')
        with zipfile.ZipFile(z_fn, 'r') as z:
            z.extractall(path=self.tmp_d_images)
            z_fns = z.namelist()
        z_fns = [x for x in z_fns if x.startswith('00')]
        
        self.z_dm_fns_ims = [x for x in z_fns if x.split('.')[-1].lower() in ['dm3', 'dm4']]
        self.z_bn_fns_ims = [x for x in z_fns if x.split('.')[-1].lower() in ['bin', 'mib']]
        self.z_hd_fns_ims = [x for x in z_fns if x.split('.')[-1].lower() in ['hdr']]
        self.z_dm_fns_ims.sort()
        self.z_bn_fns_ims.sort()
        self.z_hd_fns_ims.sort()
        
        self.ref_ims = np.load(os.path.join(self.bp, 'file_datasets',
                                            'medipix_test_binary_data_images_fpd_sum_dif_ims.npz'))
        
        # different combinations of operational mode, pixel pitch and number of active counters
        self.tmp_d_colour = tempfile.mktemp(dir=self.tmp_d, prefix='colour-')
        z_fn = os.path.join(self.bp, 'file_datasets', 'medipix_test_colour_data_v0p67p0p9.zip')
        with zipfile.ZipFile(z_fn, 'r') as z:
            z.extractall(path=self.tmp_d_colour)
            z_fns = z.namelist()   
        self.z_dm_fns_colour = [x for x in z_fns if x.split('.')[-1].lower() in ['dm3', 'dm4']]
        self.z_bn_fns_colour = [x for x in z_fns if x.split('.')[-1].lower() in ['bin', 'mib']]
        self.z_hd_fns_colour = [x for x in z_fns if x.split('.')[-1].lower() in ['hdr']]
        self.z_dm_fns_colour.sort()
        self.z_bn_fns_colour.sort()
        self.z_hd_fns_colour.sort()
        
    def convert_file(self, i, cpd={}, h5pd={}):
        #print(self.z_bn_fns[i])
        dmfns = [os.path.join(self.tmp_d, self.z_dm_fns[i])] # single file
        binfns = os.path.join(self.tmp_d, self.z_bn_fns[i])
        hdrfn = os.path.join(self.tmp_d, self.z_hd_fns[i])
        
        if 'color' in self.z_bn_fns[i]:
            # update detector size to match data size
            cpd = cpd.copy()
            cpd.update(dict(detY=128, detX=128))
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                       ds_start_skip=0, row_end_skip=1, 
                       **cpd)
        fn = mb.write_hdf5(ow=True, progress_bar=False, **h5pd)
    
    def test_convert(self):
        print(self.id())
        for i in range(len(self.z_bn_fns)):
            self.convert_file(i)
    
    def convert_file_raw(self, i, cpd={}, h5pd={}):
        #print(self.z_bn_fns[i])
        dmfns = [os.path.join(self.tmp_d_raw, self.z_dm_fns_raw[i])]
        binfns = os.path.join(self.tmp_d_raw, self.z_bn_fns_raw[i])
        hdrfn = os.path.join(self.tmp_d_raw, self.z_hd_fns_raw[i])
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                       ds_start_skip=0, row_end_skip=1, 
                       **cpd)
        fn = mb.write_hdf5(ow=True, progress_bar=False, **h5pd)
    
    def test_convert_raw(self):
        print(self.id())
        for i in range(len(self.z_bn_fns_raw)):
            self.convert_file_raw(i)    
    
    def test_convert_mi(self):
        print(self.id())
        # multiple images
        dmfns = [os.path.join(self.tmp_d_multi_ims, self.z_mi_dm_fns[0])]
        binfns = [os.path.join(self.tmp_d_multi_ims, t) for t in self.z_mi_bn_fns]
        hdrfn = os.path.join(self.tmp_d_multi_ims, self.z_mi_hd_fns[0])
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                       ds_start_skip=0, row_end_skip=1)
        fn = mb.write_hdf5(ow=True, progress_bar=False)
    
    def test_convert_images(self):
        print(self.id())
        # files with images
        for i in range(len(self.z_dm_fns_ims)):
            tag = self.z_dm_fns_ims[i].split('.')[0]
            
            dmfns = [os.path.join(self.tmp_d_images, self.z_dm_fns_ims[i])]
            binfns = os.path.join(self.tmp_d_images, self.z_bn_fns_ims[i])
            hdrfn = os.path.join(self.tmp_d_images, self.z_hd_fns_ims[i])
            
            mb = MerlinBinary(binfns, hdrfn, dmfns,
                        ds_start_skip=0, row_end_skip=1)
            fn = mb.write_hdf5(ow=True, progress_bar=False)

            ref_im = self.ref_ims[tag]
            #plt.matshow(ref_im)
            with h5py.File(os.path.join(self.tmp_d_images, tag + '.hdf5'), mode='r') as h:
                new_im = h['fpd_expt/fpd_sum_dif/data'][:]
            assert (np.isclose(np.abs(new_im-ref_im).sum(), 0))
            # isclose used in case of dtype issues
    
    def test_to_array(self):
        print(self.id())
        i = 2 # SPM mode
        dmfns = [os.path.join(self.tmp_d, self.z_dm_fns[i])]
        binfns = os.path.join(self.tmp_d, self.z_bn_fns[i])
        hdrfn = os.path.join(self.tmp_d, self.z_hd_fns[i])
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                            ds_start_skip=0, row_end_skip=1)
        print('to array')
        a = mb.to_array(progress_bar=False)
        
    def test_to_array_max(self):
        print(self.id())
        i = 2 # SPM mode
        dmfns = [os.path.join(self.tmp_d, self.z_dm_fns[i])]
        binfns = os.path.join(self.tmp_d, self.z_bn_fns[i])
        hdrfn = os.path.join(self.tmp_d, self.z_hd_fns[i])
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                            ds_start_skip=0, row_end_skip=1)
        print('to array (read_max)')
        a = mb.to_array(read_max=mb._scanX, progress_bar=False)
    
    def test_colour_conversion(self):
        for i in range(len(self.z_bn_fns_colour)):
            dmfns = []
            binfns = os.path.join(self.tmp_d_colour, self.z_bn_fns_colour[i])
            hdrfn = os.path.join(self.tmp_d_colour, self.z_hd_fns_colour[i])
            
            if i < 6 or (i > 14 and i < 21):
                # For those test datasets that were not acquired with the detector
                # operating in some kind of colour-based mode
                detYX = 256
            
            else:
                # For those test datasets that were acquired with the detector 
                # opearting in some kind of colour mode
                detYX = 128

            mb = MerlinBinary(binfns, hdrfn, dmfns, 
                            detY=detYX, detX=detYX, 
                            ds_start_skip=0, row_end_skip=0, 
                            memmap=False)
            print('to hdf5:', self.z_bn_fns_colour[i])
            a = mb.write_hdf5(ow=True, progress_bar=False)
        
    
    def test_headerless_hdf5(self):
        for i in range(len(self.z_bn_fns_colour)):
            dmfns = []
            binfns = os.path.join(self.tmp_d_colour, self.z_bn_fns_colour[i])
            hdrfn = None
            
            if i < 6 or (i > 14 and i < 21):
                # For those test datasets that were not acquired with the detector
                # operating in some kind of colour-based mode
                detYX = 256
            
            else:
                # For those test datasets that were acquired with the detector 
                # opearting in some kind of colour mode
                detYX = 128

            mb = MerlinBinary(binfns, hdrfn, dmfns, 
                            detY=detYX, detX=detYX, 
                            ds_start_skip=0, row_end_skip=0, 
                            memmap=False)
            print('to hdf5 without header:', self.z_bn_fns_colour[i])
            a = mb.write_hdf5(ow=True, progress_bar=False)
    
    def test_getitem(self):
        print(self.id())
        i = 2 # SPM mode
        dmfns = [os.path.join(self.tmp_d, self.z_dm_fns[i])]
        binfns = os.path.join(self.tmp_d, self.z_bn_fns[i])
        hdrfn = os.path.join(self.tmp_d, self.z_hd_fns[i])
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                            ds_start_skip=0, row_end_skip=1)
        print('getitem')
        a = mb[:]
        a = mb[...]
        a = mb[:, ...]
        a = mb[..., :, 0]
        a = mb[0, 0, 0, 0]
        a = mb[::2, ::1, ..., ::4, ::4]
        
        del a
        del mb
    
    def test_getitem_no_progress_bar(self):
        print(self.id())
        i = 2 # SPM mode
        dmfns = [os.path.join(self.tmp_d, self.z_dm_fns[i])]
        binfns = os.path.join(self.tmp_d, self.z_bn_fns[i])
        hdrfn = os.path.join(self.tmp_d, self.z_hd_fns[i])
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                          ds_start_skip=0, row_end_skip=1,
                          array_interface_progress_bar=False)
        print('getitem')
        a = mb[:]
        a = mb[...]
        a = mb[:, ...]
        a = mb[..., :, 0]
        a = mb[0, 0, 0, 0]
        a = mb[::2, ::1, ..., ::4, ::4]
        
        del a
        del mb
    
    def test_get_memmap(self):
        print(self.id())
        i = 2 # SPM mode
        dmfns = [os.path.join(self.tmp_d, self.z_dm_fns[i])]
        binfns = os.path.join(self.tmp_d, self.z_bn_fns[i])
        hdrfn = os.path.join(self.tmp_d, self.z_hd_fns[i])
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                            ds_start_skip=0, row_end_skip=1)
        print('getmemmap')
        mm = mb.get_memmap()
        a = mm.sum((-2, -1))
        a = mm[:]
        a = mm[...]
        a = mm[:, ...]
        a = mm[..., :, 0]
        a = mm[0, 0, 0, 0]
        a = mm[::2, ::1, ..., ::4, ::4]
        del a, mm, mb    
    
    def test_file_browser_ndarray(self):
        print(self.id())
        data = np.random.rand( *(16,)*4 )
        d = DataBrowser(data, progress_bar=False)
        plt.close('all')
    
    def test_file_browser_hdf5(self):
        print(self.id())
        data = np.random.rand( *(16,)*4 )
        
        fn = os.path.join(self.tmp_d, 'non_fpd.hdf5')
        with h5py.File(fn, mode='w') as f:
            dset = f.create_dataset("data", data=data)
            d = DataBrowser(dset, progress_bar=False)
        plt.close('all')
    
    def test_file_browser_ndarray_nav_im(self):
        print(self.id())
        data = np.random.rand( *(16,)*4 )
        mav_im = data.sum((-2, -1))
        d = DataBrowser(data, nav_im=mav_im)
        plt.close('all')
    
    def test_zForLast_file_browser(self):
        print(self.id())
        # run last for headless closing of file issues
        i = 2 # SPM mode
           
        binfn = os.path.splitext(self.z_bn_fns[i])[0]+'.hdf5'
        h5fn = os.path.join(self.tmp_d, binfn)
        if not os.path.isfile(h5fn):
            self.convert_file(i)
        d = DataBrowser(h5fn, fpd_check=True, norm='lin',)
        #d.h5f.file.close()  # for headless plotting
        plt.close('all')
    
    def test_zForLast_file_browser_kwd_dict(self):
        print(self.id())
        # run last for headless closing of file issues
        i = 2 # SPM mode
           
        binfn = os.path.splitext(self.z_bn_fns[i])[0]+'.hdf5'
        h5fn = os.path.join(self.tmp_d, binfn)
        if not os.path.isfile(h5fn):
            self.convert_file(i)
        d = DataBrowser(h5fn, fpd_check=True, norm='lin', nav_im_dict={'cmap': 'jet', 'vmin': 0, 'vmax': 1})
        #d.h5f.file.close()  # for headless plotting
        plt.close('all')
    
    def test_scanYalu(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'scanYalu.hdf5')
        scanYalu = (8, 'y-axis', 'nm')
        self.convert_file(i, h5pd={'h5fn':h5fn}, cpd={'scanYalu':scanYalu})
        
    def test_scanXalu(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'scanXalu.hdf5')
        scanXalu = (16, 'x-axis', 'nm')
        self.convert_file(i, h5pd={'h5fn':h5fn}, cpd={'scanXalu':scanXalu})
    
    def test_scanYXalu(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'scanYXalu.hdf5')
        scanYalu = (8, 'y-axis', 'nm')
        scanXalu = (16, 'x-axis', 'nm')
        self.convert_file(i, h5pd={'h5fn':h5fn}, cpd={'scanYalu':scanYalu, 'scanXalu':scanXalu})
    
    def test_scanYalu_array(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'scanYalu_array.hdf5')
        scanYalu = (np.arange(8)*0.1, 'y-axis', 'nm')
        self.convert_file(i, h5pd={'h5fn':h5fn}, cpd={'scanYalu':scanYalu})
    
    def test_scanXalu_array(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'scanXalu_array.hdf5')
        scanXalu = (np.arange(16)*0.1, 'y-axis', 'nm')
        self.convert_file(i, h5pd={'h5fn':h5fn}, cpd={'scanXalu':scanXalu})
    
    
    
    def test_detYalu(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'detYalu.hdf5')
        detYalu = (256, 'y-axis', 'nm')
        self.convert_file(i, h5pd={'h5fn':h5fn}, cpd={'detYalu':detYalu})
        
    def test_detXalu(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'detXalu.hdf5')
        detXalu = (256, 'x-axis', 'nm')
        self.convert_file(i, h5pd={'h5fn':h5fn}, cpd={'detXalu':detXalu})
    
    def test_detYXalu(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'detYXalu.hdf5')
        detYalu = (256, 'y-axis', 'nm')
        detXalu = (256, 'x-axis', 'nm')
        self.convert_file(i, h5pd={'h5fn':h5fn}, cpd={'detYalu':detYalu, 'detXalu':detXalu})
    
    def test_detYalu_array(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'detYalu_array.hdf5')
        detYalu = (np.arange(256)*0.1, 'y-axis', 'nm')
        self.convert_file(i, h5pd={'h5fn':h5fn}, cpd={'detYalu':detYalu})
    
    def test_detXalu_array(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'detXalu_array.hdf5')
        detXalu = (np.arange(256)*0.1, 'y-axis', 'nm')
        self.convert_file(i, h5pd={'h5fn':h5fn}, cpd={'detXalu':detXalu})
    
    
    
    def test_compression_opts_setting(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'compression.hdf5')
        self.convert_file(i, h5pd={'h5fn':h5fn, 'compression_opts':0})
    
    def test_mask(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'masked.hdf5')
        mask = np.eye(256,256)
        self.convert_file(i, h5pd={'h5fn':h5fn, 'mask':mask})
    
    def test_func(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'funced.hdf5')
        def my_func(image, scanYind, scanXind, my_arg):
            #print(scanYind, scanXind)
            assert(my_arg==8)
            return image + 200
        self.convert_file(i, h5pd={'h5fn':h5fn, 'func':my_func, 'func_kwargs':{'my_arg': 8}})
    
    def test_chunks_setting(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'chunks.hdf5')
        self.convert_file(i, h5pd={'h5fn':h5fn, 'chunks':(1,1,128,128)})
        
    def test_no_repacking(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'repacked.hdf5')
        self.convert_file(i, h5pd={'h5fn':h5fn, 'chunks':(8,)*4, 'repack':False})
    
    def test_chunk_coercion(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'repacked.hdf5')
        self.convert_file(i, h5pd={'h5fn':h5fn, 'chunks':(16,)*4, 'repack':True})
    
    def test_im_chunks_setting(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'im_chunks.hdf5')
        self.convert_file(i, h5pd={'h5fn':h5fn, 'im_chunks':(32, 32)})
    
    def test_scan_chunks_setting(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'scan_chunks.hdf5')
        self.convert_file(i, h5pd={'h5fn':h5fn, 'scan_chunks':(2, 2)})
    
    def test_convert_all(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'convert_all.hdf5')
        dmfns = []
        binfns = os.path.join(self.tmp_d, self.z_bn_fns[i])
        hdrfn = os.path.join(self.tmp_d, self.z_hd_fns[i])
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                          ds_start_skip=0, row_end_skip=0)
        fn = mb.write_hdf5(ow=True, progress_bar=False, h5fn=h5fn)
    
    def test_no_memmap(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'convert_all_no_memmap.hdf5')
        dmfns = []
        binfns = os.path.join(self.tmp_d, self.z_bn_fns[i])
        hdrfn = os.path.join(self.tmp_d, self.z_hd_fns[i])
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                          ds_start_skip=0, row_end_skip=0)
        fn = mb.write_hdf5(ow=True, progress_bar=False, h5fn=h5fn, allow_memmap=False)
    
    def test_LZ4(self):
        print(self.id())
        i = 2 # SPM mode
        h5fn = os.path.join(self.tmp_d, 'LZ4.hdf5')
        self.convert_file(i, h5pd={'h5fn' : h5fn, 'compression' : 'LZ4'})
        
        fpd_nt = fpdf.fpd_to_tuple(h5fn)
        d = fpd_nt.fpd_data.data[()]
        fpd_nt.file.close()
        
        fpd_signals = fpdf.fpd_to_hyperspy(h5fn, fpd_check=True, assume_yes=True)
    
    def test_tools(self):
        print(self.id())
        i = 2 # SPM mode
           
        binfn = os.path.splitext(self.z_bn_fns[i])[0]+'.hdf5'
        h5fn = os.path.join(self.tmp_d, binfn)
        # reconvert with source embedded
        self.convert_file(i, h5pd={'embed_source':True})
        
        fpg = h5fn
        isfpd, isknown, vs = fpdf._check_fpd_file(fpg, min_version=None, max_version=None)
        isfpd, isknown, vs = fpdf._check_fpd_file(fpg, min_version=None, max_version=None, silent=True)
        
        tagds, dm_group_paths, dm_filenames = fpdf.hdf5_dm_tags_to_dict(fpg, 
                                                                        fpd_check=True)
        
        out_dm_filenames = [os.path.join(self.tmp_d, t.split('.')[0]+'_extracted') for t in dm_filenames]
        fpdf.hdf5_dm_to_bin(fpg, dmfns=out_dm_filenames, fpd_check=True, ow=True)
        
        fpdf.hdf5_fpd_to_bin(fpg, fpd_fn=os.path.join(self.tmp_d, 'bin_extract.bin'),
                             fpd_check=True, ow=False)
        
        fpdf.hdf5_src_to_file(fpg, src_fn=os.path.join(self.tmp_d, 'fpd_file.py'),
                              fpd_check=True, ow=False)
        
        # hyperspy
        fpd_signals = fpdf.fpd_to_hyperspy(fpg, fpd_check=True, assume_yes=True)
        
        fpd_signals = fpdf.fpd_to_hyperspy(fpg, fpd_check=True, assume_yes=True,
                                            group_names=['fpd_data', 
                                                         'fpd_sum_im',
                                                         'fpd_sum_dif'])
        
        # namedtuple
        fpd_nt = fpdf.fpd_to_tuple(fpg, group_names='fpd_sum_im')
        fpd_nt = fpdf.fpd_to_tuple(fpg, group_names=['fpd_sum_dif', 'fpd_sum_im'])
        fpd_nt = fpdf.fpd_to_tuple(fpg, group_names=None)
        
        d = fpd_nt.fpd_data.data
        n = fpd_nt.fpd_data.name
        u = fpd_nt.fpd_data.units
        ax = fpd_nt.fpd_data.dim1
        axd = ax.data
        axn = ax.name
        axu = ax.units
        
        fpd_nt.file.close()
        
        # context manager
        with fpdf.fpd_nt_cm(fpg) as fpd_nt:
            # file is open here
            assert(bool(fpd_nt.file))
        # file is closed here
        assert(bool(fpd_nt.file) is False)
        
    
    def test_v0p67p0p8(self):
        print(self.id())
        # 1d scan of 16 images, no dm file
        
        z_fn = os.path.join(self.bp, 'file_datasets', 'medipix_test_binary_data_v0p67p0p8_small_16.tar.gz')
        
        with tarfile.open(z_fn, "r:gz") as tar:
            names = tar.getnames()
            path = names[0]
            tar.extractall(path=self.tmp_d)
        
        mibfiles = [n for n in names if n.endswith('.mib')]
        hdrfiles = [n for n in names if n.endswith('.hdr')]
        mibfiles.sort()
        hdrfiles.sort()
        
        binfns = os.path.join(self.tmp_d, mibfiles[0])
        hdrfn = os.path.join(self.tmp_d, hdrfiles[0])
        dmfns = []
        
        scanYalu = (1, 'na', 'na')
        scanXalu = (None, 'images', 'index')
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                       ds_start_skip=0, row_end_skip=0, 
                       scanYalu=scanYalu, scanXalu=scanXalu)

        fn = mb.write_hdf5(ow=True, progress_bar=False)
    
    def test_CubicImageInterpolator(self):
        print(self.id())
        yi, xi = np.indices((32, 32))
        ri = np.hypot(yi-yi.mean(), xi-xi.mean())
        image = ri.copy()
        mask = np.logical_and(ri>6, ri<8)
        image[mask] = 0

        #plt.matshow(ri)
        #plt.matshow(image)
        #plt.matshow(mask)

        CI = fpdf.CubicImageInterpolator(mask)
        image_masked = CI.interpolate_image(image, in_place=False)
        #plt.matshow(image_masked)
        #plt.matshow(image)

        assert np.isclose(ri, image_masked, rtol=1e-02, atol=0.5).all()
        assert np.isclose(ri, image, rtol=1e-02, atol=0.5).all() == False

        CI.interpolate_image(image, in_place=True)
        assert np.isclose(ri, image, rtol=1e-02, atol=0.5).all() == True
        #plt.matshow(image)
    
    
    def test_make_updated_fpd_file(self):
        print(self.id())
        # run last for headless closing of file issues
        i = 2 # SPM mode
           
        binfn = os.path.splitext(self.z_bn_fns[i])[0]+'.hdf5'
        h5fn = os.path.join(self.tmp_d, binfn)
        if not os.path.isfile(h5fn):
            self.convert_file(i)
        
        scale = 0.5
        def shift_func(image, scanYind, scanXind, scale):
            new_im = (image * scale).astype(int)
            return new_im
        
        new_fn = fpdf.make_updated_fpd_file(h5fn, os.path.join(self.tmp_d, 'mk_updated'),
                                            func=shift_func, func_kwargs=dict(scale=scale),
                                            ow=True, progress_bar=False)
        n = fpdf.fpd_to_tuple(new_fn)
        n.file.close()
    
    def test_zz_update_calibration(self):
        print(self.id())
        # use existing files        
        fn = '005_12bit_color_mode_16x8.hdf5'
        fn = os.path.join(self.tmp_d, fn)
        fpdf.update_calibration(fn, scan=[0.1, 'p'], detector=[0.2, 'q'], colour=[0.3, 'r'])
        t = fpdf.fpd_to_tuple(fn)
        assert np.diff(t.fpd_data.dim3.data)[0] == 0.3
        assert t.fpd_data.dim3.units == 'r'
        t.file.close()
        
        fn = 'masked.hdf5'
        fn = os.path.join(self.tmp_d, fn)
        fpdf.update_calibration(fn, scan=[0.1, 'p'], detector=[0.2, 'q'], colour=[None, None])
        t = fpdf.fpd_to_tuple(fn)
        assert np.diff(t.fpd_data.dim1.data)[0] == 0.1
        assert t.fpd_data.dim1.units == 'p'
        assert np.diff(t.fpd_data.dim4.data)[0] == 0.2
        assert t.fpd_data.dim3.units == 'q'
        assert np.diff(t.fpd_mask.dim1.data)[0] == 0.2
        assert t.fpd_mask.dim2.units == 'q'
        t.file.close()
    
    
    def test_v0p75p4p84(self):
        print(self.id())
        # 2d scan of 16x16 from v 0.75.4.84
                
        z_fn = os.path.join(self.bp, 'file_datasets', 'NIST_FpdTestData.tar.xz')
        
        with tarfile.open(z_fn, "r:xz") as tar:
            names = tar.getnames()
            path = names[0]
            tar.extractall(path=self.tmp_d)
        
        mibfiles = [n for n in names if n.endswith('.mib')]
        hdrfiles = [n for n in names if n.endswith('.hdr')]
        dmfiles = [n for n in names if n.endswith('.dm3')]
        mibfiles.sort()
        hdrfiles.sort()
        dmfiles.sort()
        
        binfns = os.path.join(self.tmp_d, mibfiles[0])
        hdrfn = os.path.join(self.tmp_d, hdrfiles[0])
        dmfns = os.path.join(self.tmp_d, dmfiles[0])
        
        mb = MerlinBinary(binfns, hdrfn, dmfns,
                       ds_start_skip=0, row_end_skip=0)
        fn = mb.write_hdf5(ow=True, progress_bar=False)
    
    def test_v0p77p0p16(self):
        print(self.id())
        # 2d scan of 3x3 from v 0.77.0.16
                
        z_fn = os.path.join(self.bp, 'file_datasets', 'NIST_test_data_0.77.0.16.tar.xz')
        
        with tarfile.open(z_fn, "r:xz") as tar:
            names = tar.getnames()
            path = names[0]
            tar.extractall(path=self.tmp_d)
        
        mibfiles = [n for n in names if n.endswith('.mib')]
        hdrfiles = [n for n in names if n.endswith('.hdr')]
        mibfiles.sort()
        hdrfiles.sort()
        
        binfns = os.path.join(self.tmp_d, mibfiles[0])
        hdrfn = os.path.join(self.tmp_d, hdrfiles[0])
        
        mb = MerlinBinary(binfns, hdrfn,
                       ds_start_skip=0, row_end_skip=0,
                       scanXalu=(3, '', ''), scanYalu=(3, '', ''))
        fn = mb.write_hdf5(ow=True, progress_bar=False)
    
    @classmethod
    def tearDownClass(self):
        #print(self.tmp_d)
        shutil.rmtree(self.tmp_d)
        #pass
    
    
    '''
    # for testing old files
    bd = '/media/Data/ssp-serv/fpd/merlin_test_data/new_merlin_card_20150824/'
    fns =   [u'ccsm010ms.hdr',
            u'ccsm010ms.mib',
            u'cm010ms.hdr',
            u'cm010ms.mib',
            u'csm010ms.hdr',
            u'csm010ms.mib',
            u'spm010ms.hdr',
            u'spm010ms.mib']

    hdr_fns = fns[0::2]
    bin_fns = fns[1::2]


    for h,f in zip(hdr_fns,bin_fns):
        print(os.path.join(bd,f))
        print(os.path.join(bd,h))
        binary_to_hdf5(os.path.join(bd,f), os.path.join(bd,h), ds_start_skip=0,
                    row_end_skip=0, ow=True,
                    scanYalu=(1,0,'y'), scanXalu=(1,0,'x'))

    '''


if __name__ == '__main__':
    unittest.main()
