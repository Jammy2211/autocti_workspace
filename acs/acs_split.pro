function acs_calibrated_gain,header,commanded_gain=commanded_gain,SILENT=silent
; Look up engineering calibrated CCD gain in ACS/WFC
commanded_gain=sxpar(header,"CCDGAIN")
calibrated_gain=[sxpar(header,"ATODGNA"),sxpar(header,"ATODGNB"),sxpar(header,"ATODGNC"),sxpar(header,"ATODGND")]
if min(calibrated_gain) lt 0.1 then begin
  if not keyword_set(commanded_gain) then message,"Unknown gain setting!"
  case round(commanded_gain) of 
    1: calibrated_gain=[0.99989998,0.97210002,1.01070000,1.01800000]
    2: calibrated_gain=[2.002,1.945,2.028,1.994]
    4: calibrated_gain=[4.011,3.902,4.074,3.996]
    else: message,"Unrecognised gain setting!"
  endcase
endif
message,/INFO,"Commanded gain:  "+strtrim(commanded_gain,2),NOPRINT=SILENT
message,/INFO,"Calibrated gain: "+strtrim(calibrated_gain[0],2)+" "+strtrim(calibrated_gain[1],2)+" "+strtrim(calibrated_gain[2],2)+" "+strtrim(calibrated_gain[3],2),NOPRINT=SILENT
return,calibrated_gain
end

;*****************************************************************************************************

function acs_biascorr, file, $
                       bias_file=bias_file, $
                       bias_commanded=bias_commanded, $
                       median_bias_levels=median_bias_levels, $
                       commanded_gain=commanded_gain, $
                       silent=silent

; NAME:
;      ACS_BIASCORR
;
; CATEGORY:
;      Pixel-by-pixel correction of trailing due to charge transfer 
;      inefficiency in CCD detectors.
;
; PURPOSE:
;      Obtain the bias level applied to an ACS CCD readout, in units of electrons.
;      This is merely a common subroutine shared by acs_split and acs_unsplit (below).
;
; INPUTS:
;      FILE      - String containing the absolute file name of the science image.
;
; OPTIONAL INPUTS:
;      BIAS_FILE - String containing the absolute file name of the bias image.
;                  DEFAULT: read from FITS header of science image.
;
; KEYWORD PARAMETERS:
;      BIAS_COMMA- Use the commanded (rather than measured) bias values in each CCD.
;
; OUTPUTS:
;      Returns a [2048,2048,4] array, corresponding to the bias level in each of the
;      four ACS amplifiers.
;
; OPTIONAL OUTPUTS:
;      None.
;
; MODIFICATION HISTORY:
;      Mar 08 - Written by Richard Massey.

; Read in primary FITS header, and check that all is well
if n_elements(file) ne 1 then message,"Please input only one file name at a time!"
if not file_test(file) then message,"File "+file+" not found!"
junk=mrdfits(file,0,header,/SILENT)
if strupcase(strtrim(sxpar(header,"TELESCOP"),2)) ne "HST" or $
   strupcase(strtrim(sxpar(header,"INSTRUME"),2)) ne "ACS" then message,"Instrument not recognised!"

; Determine bias level
if keyword_set(bias_commanded) then begin
  message,"Using commanded value of residual bias.",/INFO,NOPRINT=silent
  bias_image=[[[replicate(sxpar(header,"CCDOFSTA"),2072,2068)]], $
              [[replicate(sxpar(header,"CCDOFSTB"),2072,2068)]], $
              [[replicate(sxpar(header,"CCDOFSTC"),2072,2068)]], $
              [[replicate(sxpar(header,"CCDOFSTD"),2072,2068)]]]
endif else begin
  if not keyword_set(bias_file) then begin
    bias_file=sxpar(header,"BIASFILE")
    jref=strpos(bias_file,"jref$") & if jref ge 0 then bias_file=file_dirname(file,/MARK_DIRECTORY)+strmid(bias_file,jref+5)
  endif
  ;if not file_test(bias_file) then begin
  ;  print,"Trying to download bias file "+file_basename(bias_file)
  ;  spawn,"wget -O "+bias_file+" https://hst-crds.stsci.edu/unchecked_get/references/hst/"+file_basename(bias_file)
  ;endif
  if file_test(bias_file) then begin
    message,"Reading residual bias image "+bias_file,/INFO,NOPRINT=silent
    bias_image_ccd1=reverse(mrdfits(bias_file,4,bias_header_ccd1,/SILENT),2)
    bias_image_ccd2=        mrdfits(bias_file,1,bias_header_ccd2,/SILENT)
    bscale=sxpar(bias_header_ccd1,"BSCALE") & bzero=sxpar(bias_header_ccd1,"BZERO")
    if bscale ne 0 then bias_image_ccd1=temporary(bias_image_ccd1)*bscale
    bias_image_ccd1=temporary(bias_image_ccd1)+bzero
    bscale=sxpar(bias_header_ccd2,"BSCALE") & bzero=sxpar(bias_header_ccd2,"BZERO")
    if bscale ne 0 then bias_image_ccd2=temporary(bias_image_ccd2)*bscale
    bias_image_ccd2=temporary(bias_image_ccd2)+bzero
    ltv1=round(sxpar(bias_header_ccd1,"LTV1"))
;    bias_image=[[[(reverse(bias_image_ccd1,1))[ltv1:ltv1+2047,0:2047]]],$
;                [[bias_image_ccd1[ltv1:ltv1+2047,0:2047]]],$
;                [[bias_image_ccd2[ltv1:ltv1+2047,0:2047]]],$
;                [[(reverse(bias_image_ccd2,1))[ltv1:ltv1+2047,0:2047]]]]
    bias_image=[[[(reverse(bias_image_ccd1,1))[0:ltv1+2047,*]]],$
                [[bias_image_ccd1[0:ltv1+2047,*]]],$
                [[bias_image_ccd2[0:ltv1+2047,*]]],$
                [[(reverse(bias_image_ccd2,1))[0:ltv1+2047,*]]]]
    case strupcase(strtrim(sxpar(bias_header_ccd1,"BUNIT"),2)) of
      "ELECTRONS":
      "COUNTS": begin
                  ;bias_image=bias_image*(sxpar(header,"CCDGAIN")>1); in v1.0
                  calibrated_gain=acs_calibrated_gain(header,commanded_gain=commanded_gain)
                  for k=0,3 do bias_image[*,*,k]=bias_image[*,*,k]*calibrated_gain[k]
                end
      else: message,"Image units not recognised!"
    endcase
  endif else message,"File "+bias_file+" not found!"
endelse

; Report back to outside world
median_bias_levels=[median(bias_image[*,*,0]),median(bias_image[*,*,1]),median(bias_image[*,*,2]),median(bias_image[*,*,3])]
message,"Median residual bias:"+strcompress(strjoin(median_bias_levels)),/INFO,NOPRINT=silent
return,bias_image

end

;*****************************************************************************************************

function acs_blevcorr, image, $
                       bias_file=bias_file, $
                       blev_linear=blev_linear, $
                       blev_params=blev_params, $
                       silent=silent

; NAME:
;      ACS_BLEVCORR
;
; CATEGORY:
;      Pixel-by-pixel correction of trailing due to charge transfer 
;      inefficiency in CCD detectors.
;
; PURPOSE:
;      Obtain the bias level applied to an ACS CCD readout, in units of electrons.
;      This is merely a common subroutine shared by acs_split and acs_unsplit (below).
;
; INPUTS:
;      FILE      - String containing the absolute file name of the science image.
;
; OPTIONAL INPUTS:
;      BIAS_FILE - String containing the absolute file name of the bias image.
;                  DEFAULT: read from FITS header of science image.
;
; KEYWORD PARAMETERS:
;      BLEV_LINEA- Allow linear variation in the bias level.
;
; OUTPUTS:
;      Returns a [2048,2048,4] array, corresponding to the bias level in each of the
;      four ACS amplifiers.
;
; OPTIONAL OUTPUTS:
;      None.
;
; MODIFICATION HISTORY:
;      Mar 08 - Written by Richard Massey.

; Determine bias level
if not keyword_set(blev_params) then begin
  blev_params=fltarr(4,2)
  message,"Measuring bias level from overscan regions.",/INFO,NOPRINT=silent
  for i=0,3 do begin
    overscan=reform(image[18:23,0:2047,i],6*2048)
    if keyword_set(blev_linear) then begin
      ; notation from Anton p.461
      M=[[replicate(1,6*2048)],[reform(replicate(1.,6)#findgen(2048),6*2048)]]
      blev_params[i,*]=invert(transpose(M)#M)##(M##overscan)
    endif else blev_params[i,0]=mean(overscan)
  endfor
endif

; Create bias "image"
y=replicate(1.,2072)#findgen(2068)
;bias_image=make_array(2048,2048,4,type=size(image,/TYPE)) & for j=0,3 do bias_image[*,*,j]=median(overscan[*,*,j])
bias_image=fltarr(2072,2068,4)
for i=0,3 do bias_image[*,*,i]=blev_params[i,0]+blev_params[i,1]*y

; Report back to outside world
message,"O/scan bias levels:    "+strcompress(strjoin(blev_params[*,0])),/INFO,NOPRINT=silent
message,"O/scan bias gradients: "+strcompress(strjoin(blev_params[*,1])),/INFO,NOPRINT=silent
return,bias_image

end

;*****************************************************************************************************

pro acs_unsplit, files,                                 $
                 suffix=suffix,                         $
                 extension=extension,                   $
                 exposure_time=exposure_time,           $
                 no_bias_correction=no_bias_correction, $
                 bias_file=bias_file,                   $
                 bias_commanded=bias_commanded,         $
                 blev_linear=blev_linear,               $
                 silent=silent

; NAME:
;      ACS_UNSPLIT
;
; CATEGORY:
;      Pixel-by-pixel correction of trailing due to charge transfer 
;      inefficiency in CCD detectors.
;
; PURPOSE:
;      Combine four images, each correspinding to one of the ACS WFC amplifiers,
;      into one file, based on a template file. Undoes the operation of acs_split.
;
;      For example, a _raw.fits image can be split up and the four amplifiers 
;      corrected separately for CTE trailing. Then the four _cte?.fits images
;      can be recombined into _cte.fits, using _raw.fits as a template.
;
; EXAMPLE USE:
;      acs_unsplit,"~/data/j8xib0kaq"
;
; INPUTS:
;      FILE      - String containing the absolute file name of the science image.
;
; OPTIONAL INPUTS:
;      BIAS_FILE - Full path to bias image. DEFAULT: read from FITS header.
;      BIAS_IMAGE- Bias level, as a [2048,2048,4] image array, and for internal use.
;      EXTENSION - File name extension. DEFAULT: "fits"
;      SUFFIX    - An array of two strings, the first being the original (template)
;                  suffix, and the second being the new name, used for the four images.
;                  DEFAULT: ["_raw","_cte"]
;
; KEYWORD PARAMETERS:
;      BIAS_MEDIA- Use a flat bias frame, calculated from the median in each CCD.
;      BIAS_COMMA- Use the commanded (rather than measured) bias values in each CCD.
;      BIAS_OVERS- Estimate the bias level from 
;      SILENT    - Operate without feedback.
;
; OUTPUTS:
;      Four images are written to disc, each containing the data read through
;      one amplifier.
;
; OPTIONAL OUTPUTS:
;      IMAGE     - Image array containing pixel data in units of electrons. 
;                  Regions read through different amplifiers are stacked in the 
;                  third dimension. The amplifier is at (0,0), with the parallel 
;                  direction in the second dimension.
;      HEADER    - Returns the header from unit 0 of the science FITS file.
;
; MODIFICATION HISTORY:
;      Oct 09 - Single-precision wraparound in output images fixed by RM.
;      Oct 09 - Amps A and B had been swapped (self-consistently). Corrected by RM.
;      Feb 08 - Written by Richard Massey.

; Parse inputs
if not keyword_set(suffix) then suffix=["_raw","_cte"]
if not keyword_set(extension) then extension="fits"
header_title=[string(replicate(32b,80)),"              / CTE CORRECTION MODEL PARAMETERS                                 ",string(replicate(32b,80))]

; Look at each image in turn
n_files=n_elements(files)
for i=0,n_files-1 do begin

  ; Specify the files that will be needed, and check that they exist
  output_file=files[i]
  input_file=file_dirname(files[i],/MARK_DIRECTORY)+file_basename(files[i],"."+extension,/FOLD_CASE)+["A","B","C","D"]+"."+extension
  for j=0,3 do if not file_test(input_file[j]) then message,"File "+input_file[j]+" not found!"
  template_file=file_dirname(files[i],/MARK_DIRECTORY)+file_basename(files[i],suffix[1]+"."+extension,/FOLD_CASE)+suffix[0]+"."+extension
  if not file_test(template_file) then message,"Template file "+template_file+" not found!"
  message,/INFO,(n_files le 2)?"Creating image "+output_file:"Creating image "+output_file+" ("+strtrim(round(i*100./n_files),2)+"%)",NOPRINT=silent
  message,/INFO,"Template image "+template_file,NOPRINT=silent

  ; Read in primary FITS header of template file
  header=[headfits(template_file,exten=0),headfits(template_file,exten=1)]
  if strupcase(strtrim(sxpar(header,"TELESCOP"),2)) ne "HST" or $
     strupcase(strtrim(sxpar(header,"INSTRUME"),2)) ne "ACS" then message,"Instrument not recognised!"

  ; Read in (new) data
  message,/INFO,"Reading image "+input_file[0],NOPRINT=silent & fits_read,input_file[0],imageA,headerA ; This internally corrects for any bscale or bzero
  message,/INFO,"Reading image "+input_file[1],NOPRINT=silent & fits_read,input_file[1],imageB,headerB
  message,/INFO,"Reading image "+input_file[2],NOPRINT=silent & fits_read,input_file[2],imageC,headerC
  message,/INFO,"Reading image "+input_file[3],NOPRINT=silent & fits_read,input_file[3],imageD,headerD
  input_image=[[[imageA]],[[imageB]],[[imageC]],[[imageD]]]
  blev_params=[[sxpar(headerA,"BLEV"),    sxpar(headerB,"BLEV"),    sxpar(headerC,"BLEV"),    sxpar(headerD,"BLEV")],$
               [sxpar(headerA,"BLEVGRAD"),sxpar(headerB,"BLEVGRAD"),sxpar(headerC,"BLEVGRAD"),sxpar(headerD,"BLEVGRAD")]]
  delvarx,imageA,imageB,imageC,imageD
  
  ; Add back amplifier bias
  if not keyword_set(no_bias_correction) then begin
    input_image=input_image+acs_blevcorr(input_image,blev_linear=blev_linear,blev_params=blev_params)
    input_image=temporary(input_image)+acs_biascorr(template_file,bias_file=bias_file,bias_commanded=bias_commanded,silent=silent)
  endif

  ; Convert units back to ADU if required
  case strupcase(strtrim(sxpar(header,"BUNIT"),2)) of
    "ELECTRONS": 
    "ELECTRONS/S": begin
                     no_round=1B
                     if keyword_set(exposure_time) then input_image=temporary(input_image)/exposure_time else message,"Need to know exposure time of original images!"
                   end
    "COUNTS": begin
                calibrated_gain=acs_calibrated_gain(header)
                for k=0,3 do input_image[*,*,k]=input_image[*,*,k]/calibrated_gain[k]
              end
    else: message,"Image units '"+strupcase(strtrim(sxpar(header,"BUNIT"),2))+"' not recognised!"
  endcase
  
  ; Update one data unit at a time
  for j=0,6 do begin
  
    ; Read in template data
    image=mrdfits(template_file,j,header,/SILENT)
    
    bitpix=sxpar(header,"BITPIX") & if bitpix ne 16 then message,/INFO,"Unexpected number of bits per pixel in template image"
    bzero=sxpar(header,"BZERO")
    bscale=float(sxpar(header,"BSCALE")) & if bscale eq 0 then bscale=1.
    image[*,*]=image[*,*]*bscale+bzero
    ltv1=round(sxpar(header,"LTV1"))

    ; Update data (without changing the data type) and header
    if j eq 1 or j eq 4 then begin
      if j eq 1 then begin
        image=([input_image[*,*,2],reverse(input_image[*,*,3],1)])
      endif else begin
       ;image=([input_image[*,*,1],reverse(input_image[*,*,0],1)]) ; Swapping A and B as noticed by Pey-Lian
        image=([input_image[*,*,0],reverse(input_image[*,*,1],1)])
        image=reverse(image,2)
      endelse
      header_point=max(where(strcmp(header,"END") eq 0))
      header_cte=where(strmatch(headerC,"CTE_*"),n_header_cte)
      if n_header_cte gt 0 then header=[header[0:header_point-1],header_title,headerA[header_cte],header[header_point]]
      if not keyword_set(no_round) then begin
        ; The following line used to be with the bias correction
        input_image=-0.49>temporary(input_image)<(2.^16-0.49) ; Prevent overflows in the FITS image that, with BITPIX=16, cannot handle numbers outside this range
        image=fix(round(((image-bzero)/bscale)<((2.^15)-1))) ; if bscale<1, this may cause overflow issues in the fits file
      endif
    endif
    
    ; Write out new data unit
    message,/INFO,"Writing data unit "+strtrim(j,2)+" to "+output_file,NOPRINT=silent
    mwrfits,image,output_file,header,create=(j eq 0);,iscale=1;,bscale=float(bscale);,bzero=bzero

  endfor
  message,/INFO,"Finished writing image "+output_file,NOPRINT=silent

  ;; Delete original files to conserve disk space and tidy up
  ;message,/INFO,"Deleting separate files"
  ;file_delete,input_file

endfor

end

;*****************************************************************************************************

pro acs_split, files,                                 $
               image=image,                           $
               header=header,                         $
               unsplit=unsplit,                       $
               suffix=suffix,                         $
               extension=extension,                   $
               date_obs=date_obs,                     $
               exposure_time=exposure_time,           $
               no_bias_correction=no_bias_correction, $
               bias_file=bias_file,                   $
               bias_commanded=bias_commanded,         $
               blev_linear=blev_linear,               $
               stats=stats,                           $
               silent=silent

;+
; NAME:
;      ACS_SPLIT
;
; CATEGORY:
;      Pixel-by-pixel correction of trailing due to charge transfer 
;      inefficiency in CCD detectors.
;
; PURPOSE:
;      Read in a raw image, convert it to units of electrons and split it into 
;      regions read though different amplifiers. Then write individual files
;      back to disc.
;
;      Several steps could probably have been done better in CALACS, but that 
;      requires fucking IRAF. Just don't use this routine blindly for any data
;      that is not from ACS and is not read out with all four amplifiers, etc.
;
; EXAMPLE USE:
;      acs_split,"~/data/j8xib0kaq_raw.fits"
;      acs_split,"~/data/j8xib0kaq_cte.fits",/UNSPLIT
;
; INPUTS:
;      FILE      - String containing the absolute file name of the science image.
;
; OPTIONAL INPUTS:
;      BIAS_FILE - Full path to bias image. DEFAULT: read from FITS header.
;      BIAS_IMAGE- Bias level, as a [2048,2048,4] image array, and for internal use.
;      EXTENSION - File name extension. DEFAULT: "fits"
;      SUFFIX    - Not used.
;
; KEYWORD PARAMETERS:
;      UNSPLIT   - Do the inverse operation. This just calls the ACS_UNSPLIT routine,
;                  but is included as an option here so that defienitely gets compiled.
;      BIAS_COMMA- Use the commanded (rather than measured image) bias in each CCD.
;      BLEV_LINEAR-Fit a gradient to the bias level in the images. DEFAULT: constant.
;      SILENT    - Operate without feedback.
;
; OUTPUTS:
;      Four images are written to disc, each containing the data read through
;      one amplifier. These are oriented with the parallel readout in the y direction,
;      the serial readout in the x direction, and the amplifier and the bottom-left.
;
; OPTIONAL OUTPUTS:
;      IMAGE     - Image array containing pixel data in units of electrons. 
;                  Regions read through different amplifiers are stacked in the 
;                  third dimension. The amplifier is at (0,0), with the parallel 
;                  direction in the second dimension.
;      HEADER    - Returns the header from unit 0 of the science FITS file.
;
; MODIFICATION HISTORY:
;      Oct 09 - Amps A and B had been swapped (self-consistently). Corrected by RM.
;      Feb 08 - Written by Richard Massey.
;-

; Allow acs_unsplit to be called by this routine, for convenience of compiling
if keyword_set(unsplit) then begin
  acs_unsplit, files,                                 $
               suffix=suffix,                         $
               extension=extension,                   $
               exposure_time=exposure_time,           $
               no_bias_correction=no_bias_correction, $
               bias_file=bias_file,                   $
               bias_commanded=bias_commanded,         $
               blev_linear=blev_linear,               $
               silent=silent
  return
endif

; Set up global variables to store statistics of the files
n_files=n_elements(files)
stats=replicate({pi:"",pid:0,date:0.,background:fltarr(4),gain:0,biasgain:0,bias:fltarr(4),blev:fltarr(4,2),ltv1:0,bscale:fltarr(2),bzero:fltarr(2),bunit:""},n_files)

; Look at each image in turn
if not keyword_set(extension) then extension="fits"
for i=0,n_files-1 do begin

  ; Check that the supplied file really is ACS data
  if not file_test(files[i]) then message,"File "+files[i]+" not found!"
  message,/info,(n_files le 2)?"Reading image "+files[i]:"Reading image "+files[i]+" ("+strtrim(round(i*100./n_files),2)+"%)",NOPRINT=silent
  image_temp=mrdfits(files[i],0,header,/SILENT)
  telescope=strtrim(sxpar(header,"TELESCOP"),2)
  instrument=strtrim(sxpar(header,"INSTRUME"),2)
  if telescope ne "HST" or instrument ne "ACS" then message,"Instrument not recognised!"
  stats(i).pi=strtrim(sxpar(header,"PR_INV_F"),2)+" "+strtrim(sxpar(header,"PR_INV_L"),2)
  stats(i).pid=sxpar(header,"PROPOSID")
  
  ; Read in image00
  image_ccd1=reverse(mrdfits(files[i],4,header_ccd1,/SILENT),2) ; Counterintuitively, WFC1 is stored second in the FITS file
  stats(i).bscale[0]=sxpar(header_ccd1,"BSCALE") & stats(i).bzero[0]=sxpar(header_ccd1,"BZERO")
  if stats(i).bscale[0] ne 0 then image_ccd1=temporary(image_ccd1)*stats(i).bscale[0]
  image_ccd1=temporary(image_ccd1)+stats(i).bzero[0]
  image_ccd2=mrdfits(files[i],1,header_ccd2,/SILENT)
  stats(i).bscale[1]=sxpar(header_ccd2,"BSCALE") & stats(i).bzero[1]=sxpar(header_ccd2,"BZERO")
  if stats(i).bscale[1] ne 0 then image_ccd2=temporary(image_ccd2)*stats(i).bscale[1]
  image_ccd2=temporary(image_ccd2)+stats(i).bzero[1]
  stats(i).ltv1=round(sxpar(header_ccd2,"LTV1")) ; this had better be 24
  ;image=[[[(reverse(image_ccd1,1))[0:ltv1+2047,*]]],$ ; Swapping A and B as noticed by Pey-Lian
  ;       [[image_ccd1[0:ltv1+2047,*]]],$
  ;       [[image_ccd2[0:ltv1+2047,*]]],$
  ;       [[(reverse(image_ccd2,1))[0:ltv1+2047,*]]]]
  image=[[[image_ccd1[0:stats(i).ltv1+2047,*]]],$
         [[(reverse(image_ccd1,1))[0:stats(i).ltv1+2047,*]]],$
         [[image_ccd2[0:stats(i).ltv1+2047,*]]],$
         [[(reverse(image_ccd2,1))[0:stats(i).ltv1+2047,*]]]]

  ; Convert units to electrons (gain is in units of e-/ADU)
  stats(i).bunit=strupcase(strtrim(sxpar(header_ccd1,"BUNIT"),2))
  case stats(i).bunit of
    "ELECTRONS": 
    "ELECTRONS/S": if keyword_set(exposure_time) then image=temporary(image)*exposure_time else message,"Need to know exposure time of original images!"
    "COUNTS": begin
                calibrated_gain=acs_calibrated_gain(header,commanded_gain=commanded_gain)
                for j=0,3 do image[*,*,j]=image[*,*,j]*calibrated_gain[j]
                stats(i).gain=commanded_gain 
              end
    else: print,header;message,"Image units '"+strupcase(strtrim(sxpar(header,"BUNIT"),2))+"' not recognised!"
  endcase
  
  
  
  if not keyword_set(no_bias_correction) then begin
  
    ; Determine and subtract bias level (CALACS blevcorr command)
    image=image-acs_blevcorr(image,blev_linear=blev_linear,blev_params=blev_params)
    stats(i).blev=blev_params
    
    ; Determine and subtract remaining bias image (CALACS biascorr command)
    image=temporary(image)-acs_biascorr(files[i],bias_file=bias_file,bias_commanded=bias_commanded,median_bias=median_bias,commanded_gain=commanded_gain,silent=silent)
    stats(i).bias=median_bias
    stats(i).biasgain=commanded_gain

  endif

  ;message,/info,'does it matter if the split images are negative?'
  ; Convert back to integers, since that is all that makes sense (must have a larger range than fix, since typically multiple electrons per ADU)
  ;image=long(temporary(image>0)+0.49)
  
  ; Determine date of observation
  launch_date=2452334.5
  time_obs=[0,0,0]
  if not keyword_set(date_obs) then begin
    date_obs=strsplit(sxpar(header,"DATE-OBS"),"-",/extract)
    if date_obs[0] eq 0 then date_obs=strsplit(sxpar(header,"DATE"),"-",/extract)
    time_obs=strsplit(sxpar(header,"TIME-OBS"),":",/extract)
    if time_obs[0] eq 0 then time_obs=[0,0,0]
  endif else time_obs=[0,0,0]
  jdcnv,date_obs[0],date_obs[1],date_obs[2],time_obs[0]+time_obs[1]/60.+time_obs[2]/3600.,jd
  stats(i).date=jd-launch_date;2400000.5
 
  ; Write images of each amplifier to separate files, including the date information in their headers
  mkhdr, new_header, image[*,*,0]
  sxaddpar, new_header, "BUNIT", "ELECTRONS"
  sxaddpar, new_header, "BSCALE", 1
  sxaddpar, new_header, "BZERO", 0
  sxaddpar, new_header, "DATE", stats(i).date, "Days since launch when image was acquired"
  for j=0,3 do begin
    amplifier=(["A","B","C","D"])[j]
    sxaddpar,new_header,"WFC_AMP",amplifier ; This overwrites the old value if it's already there
    sxaddpar,new_header,"BLEV",stats(i).blev[j,0]
    sxaddpar,new_header,"BLEVGRAD",stats(i).blev[j,1]
    file_new=file_dirname(files[i],/MARK_DIRECTORY)+file_basename(files[i],"."+extension,/FOLD_CASE)+amplifier+"."+extension
    message,"Writing image "+file_new,/INFO,NOPRINT=silent
    fits_write,file_new,image[*,*,j],new_header
  endfor

endfor

end
