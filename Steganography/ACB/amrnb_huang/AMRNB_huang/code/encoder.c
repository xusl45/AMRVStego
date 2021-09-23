// coder.cpp : 定义控制台应用程序的入口点。
//
 

/*
 * ===================================================================
 *  TS 26.104
 *  REL-5 V5.4.0 2004-03
 *  REL-6 V6.1.0 2004-03
 *  3GPP AMR Floating-point Speech Codec
 * ===================================================================
 *
 */

/*
 * encoder.c
 *
 *
 * Project:
 *    AMR Floating-Point Codec
 *
 * Contains:
 *    Speech encoder main program
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include "typedef.h"
#include "interf_enc.h"
#include <time.h>

#define MAXLINE 3366
#define BUFLEN 3366
#ifndef ETSI
#ifndef IF2
#define AMR_MAGIC_NUMBER "#!AMR\n"
#endif
#endif
static const short modeConv[]={
   475, 515, 59, 67, 74, 795, 102, 122};

static void Usage(char* argv[])
{
   fprintf (stderr,
      "Usage of %s:\n\n"
      "[-dtx] mode speech_file bitstream_file \n\n"
      "or \n\n"
      "[-dtx] -modefile=mode_file speech_file bitstream_file \n\n\n"
      "mode = MR475, MR515, MR59, MR67, MR74, MR795, MR102, MR122 \n\n\n",
      argv[0]);
}
void Copyright(void){
fprintf (stderr,
"===================================================================\n"
" TS 26.104                                                         \n"
" REL-5 V5.4.0 2004-03                                              \n"
" REL-6 V6.1.0 2004-03                                              \n"
" 3GPP AMR Floating-point Speech Encoder                            \n"
"===================================================================\n"
);
}



int main()
{
	int filenu,j;int em_rate;
	char Infile[100]={0};
	char Outfile[100]={0};
	int emr_real;//实际嵌入率
	//int change_subframes;
	int printsecret = 0;//是否输出秘密信息,0不,1打印,2输出txt.
	int printpra = 0;//是否输出参数,0不,1打印,2输出txt.
	//嵌入率=总的嵌入比特/总的嵌入容量(比特数)=总的嵌入比特/(总共嵌入帧数*每帧嵌入比特)
	//最大可嵌入比特数 = 嵌入率 * 总共嵌入帧数*每帧最大嵌入比特
	int total_create = 0;
    for (float time = 0.1; time < 0.99; time = time + 0.1)
    {
        int total_subframes = time * 50 * 4;//20ms一帧,1秒50帧,1帧有4整数用于嵌入


        for (int em_rate = 0; em_rate <= 100; em_rate = em_rate + 100) {
            for (filenu = 0; filenu < 8000; filenu++)//样本个数
            {
                emr_real = 0;
                sprintf(Infile, "D:\\paper1Data\\NB\\huang\\%.1fs\\%d\\hiding\\hunag_sample%04d.pcm", time, em_rate, filenu);

                sprintf(Outfile, "D:\\paper1Data\\NB\\huang\\%.1fs\\%d\\rendat\\c_sample%04d.dat", time,em_rate, filenu);
                //sprintf(Outfile, "Z:/AMR_NB_STEGO/huang/%ds/%d/sample%04d.dat", time, em_rate,filenu);
                do
                {
                    printf("--------------------------------------------\n%s\n%s\n", Infile, Outfile);
                    int change_subframes = 0;//实际修改子帧数
                    coder(Infile, Outfile, em_rate, total_subframes, filenu, &change_subframes, printsecret, printpra);
                    emr_real = 100 * change_subframes / total_subframes;//实际嵌入率
                    printf("嵌入率:%d,实际嵌入率:%d,改变子帧数:%d/%d\n", em_rate, emr_real, change_subframes, total_subframes);
                    total_create++;
                } while (emr_real < em_rate - 5);//达到指定嵌入率-5内就是合格样本,否则重新生成
            }
        }
    }
	printf("总共生成样本数量:%d", total_create);
	return 0;

}
/*
 * main
 *
 *
 * Function:
 *    Speech encoder main program
 *		语音编码器主程序
 *    Usage: encoder speech_file bitstream_file mode dtx mode_file
 *	
 *    Format for speech_file: 
 *       Speech is read from a binary file of 16 bits data.
 *
 *    Format for ETSI bitstream file:
 *       1 word (2-byte) for the TX frame type
 *       244 words (2-byte) containing 244 bits.
 *          Bit 0 = 0x0000 and Bit 1 = 0x0001
 *       1 word (2-byte) for the mode indication
 *       4 words for future use, currently written as zero
 *
 *    Format for 3GPP bitstream file:
 *       Holds mode information and bits packed to octets.
 *       Size is from 1 byte to 31 bytes.
 *
 *    ETSI bitstream file format is defined using ETSI as preprocessor
 *    definition
 *
 *    mode        : MR475, MR515, MR59, MR67, MR74, MR795, MR102, MR122
 *    mode_file   : reads mode information from a file
 * Returns:
 *    0
 */
int coder (char *infilename, char *outfilename,int em_rate,int total_subframes,int filenu,int *change_subframes, int printsecret, int printpra){
   
	int argc;
	char **argv=(char**)malloc(4*sizeof(char*));
//int main(int argc,char *argv[])
//{

   /* file strucrures */
   FILE * file_speech = NULL;
   FILE * file_encoded = NULL;
   FILE * file_mode = NULL;
    //新建lag文件

   FILE * file_lag;
     	
   /* input speech vector */
   short speech[160];

   /* counters */
   int byte_counter, frames = 0, bytes = 0;
   
   /* pointer to encoder state structure */
   int *enstate;

   /* requested mode */
   enum Mode req_mode = MR122;
   int dtx = 0;

   /* temporary variables */
   char mode_string[9];
   long mode_tmp;

   /* bitstream filetype */
#ifndef ETSI
   unsigned char serial_data[32];
#else
   short serial_data[250] = {0};
#endif

   /* Process command line options */

   argc=4;
   argv[3]=outfilename;
   argv[2]=infilename;
 // argv[3]="C:\\Users\\Administrator\\Desktop\\sample000.dat";
 // argv[2]="C:\\Users\\Administrator\\Desktop\\sample000.pcm";
   argv[1]="MR122";

   if ((argc == 5) || (argc == 4)){
      file_encoded = fopen(argv[argc - 1], "wb");
      if (file_encoded == NULL){
		 printf("dat 目录不存在\n");
         Usage(argv);
         return 1;
      }
      file_speech = fopen(argv[argc - 2], "rb");
      if (file_speech == NULL){
         fclose(file_encoded);
        //Usage(argv);
		 printf("rb %s\n",argv[argc - 2]);
         return 1;
      }

      if (strncmp(argv[argc - 3], "-modefile=", 10) == 0){
         file_mode = fopen(&argv[argc - 3][10], "rt");
         if (file_mode == NULL){
            //Usage(argv);
			printf("rt\n");
            fclose(file_speech);
            fclose(file_encoded);
            return 1;
         }
      }
      else {
         mode_tmp = strtol(&argv[argc - 3][2], NULL, 0);
         for (req_mode = 0; req_mode < 8; req_mode++){
            if (mode_tmp == modeConv[req_mode])
               break;
         }
         if (req_mode == 8){ 
            //Usage(argv); 
			printf("d\n");
            fclose(file_speech);
            fclose(file_encoded);
            if (file_mode != NULL)
               fclose(file_mode);
            return 1;
         }
      }
      if (argc == 5){
         if ((strcmp(argv[1], "-dtx") != 0)){
            //Usage(argv);
			printf("e\n");
            fclose(file_speech);
            fclose(file_encoded);
            if (file_mode != NULL){
               fclose(file_mode);
            }
            return 1;
         }
         else {
            dtx = 1;
         }
      }
   }
   else {
      //Usage(argv);
      return 1;
   }


   enstate = Encoder_Interface_init(dtx);

   //Copyright();
#ifndef VAD2
   //fprintf( stderr, "%s\n", "Code compiled with VAD option: VAD1");
#else
   fprintf( stderr, "%s\n", "Code compiled with VAD option: VAD2");
#endif

#ifndef ETSI
#ifndef IF2
   /* write magic number to indicate single channel AMR file storage format */
   	bytes = fwrite(AMR_MAGIC_NUMBER, sizeof(char), strlen(AMR_MAGIC_NUMBER), file_encoded);
#endif
#endif

   /* read file */
   while (fread( speech, sizeof (Word16), 160, file_speech ) > 0)
   {
      /* read mode */
      if (file_mode != NULL){
         req_mode = 8;
         if (fscanf(file_mode, "%9s\n", mode_string) != EOF) {
            mode_tmp = strtol(&mode_string[2], NULL, 0);
            for (req_mode = 0; req_mode < 8; req_mode++){
               if (mode_tmp == modeConv[req_mode]){
                  break;
               }
            }
         }
         if (req_mode == 8){
            break;
         }
      }

      frames ++;

      /* call encoder */
      byte_counter = Encoder_Interface_Encode(enstate, req_mode, speech, serial_data, 0, em_rate,total_subframes,filenu, change_subframes, printsecret, printpra);
	 
	  //byte_counter = Encoder_Interface_Encode(enstate, req_mode, speech, serial_data, 0);
      bytes += byte_counter;
      fwrite(serial_data, sizeof (UWord8), byte_counter, file_encoded );
      fflush(file_encoded);
   }
   Encoder_Interface_exit(enstate);


  // printf("****************************************");
  // file_lag=fopen("C:\\Users\\Administrator\\Desktop\\lag_o.txt","a");
  // for (i = 0; i < 2000; i++)
	 //  {
		//   //printf("%d\n", pitchDelay[i]);
		//   fprintf(file_lag,"%d\n",pitchDelay[i]);
		//}




#ifndef ETSI
#ifdef IF2
   fprintf ( stderr, "\n%s%i%s%i%s\n", "Frame structure AMR IF2: ", frames, " frames, ", bytes, " bytes.");
#else
   fprintf ( stderr, "%s%i%s%i%s\n", "Frame structure AMR MIME file storage format: ", frames, " frames, ", bytes, " bytes.");
#endif
#else
   fprintf ( stderr, "\n%s%i%s\n", "Frame structure AMR ETSI: ", frames, " frames. ");
#endif

   fclose(file_speech);
   fclose(file_encoded);
   if (file_mode != NULL)
      fclose(file_mode);

   return 0;
}
