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

#include <string.h>
#include <io.h>


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

const char inPATH[200] = { "I:\\MyTest\\AMR_US\\original_samples\\"};
const char fileCate[2][200] = { "测试集\\", "训练集\\"};

const char outPATH[200] = { "I:\\MyTest\\AMR_US\\FCB\\Miao\\"};

//判断文件是否存在，存在返回true，否则返回false
int file_exists(char * fileName) 
{
	return (access(fileName, 0) == 0);
}

int main(int argc, char *argv[] ) 
{

	char inFileName[200]={0};
	char outFileName[200]={0};
	int filenu,j;
	int em_rate=0;
	int enta[3]={1,2,4};//1,2,4


	for(j = 0; j < 3; j++)//enta[3]={1,2,4}
	{
		for(em_rate = 5; em_rate <= 100; em_rate += 5)
		{
			for(filenu=0;filenu<3367;filenu++)//混合样本有3367个
			{			
		sprintf(inFileName,"F:\\sample\\original_pcmfiles\\all\\sample%04d.pcm",filenu);
		sprintf(outFileName,"D:\\data\\AMR_NB_FCB_MIAO\\3367\\10s\\%d\\n%d\\sample%04d.dat",
			em_rate,enta[j],filenu);
		printf("%s\n%s\n",inFileName,outFileName);
		AMRcoder(inFileName, outFileName,em_rate, enta[j]); 
			}

		}
	}

		
	}	








/*
 * main
 *
 *
 * Function:
 *    Speech encoder main program
 *
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
int AMRcoder (char * inFileName, char * outFileName, int em_rate,int enta){
   
   /*int argc; char ** argv;*/

   int argc;
   char ** argv = (char**) malloc(4 * sizeof(char *));

   /* file strucrures */
   FILE * file_speech = NULL;
   FILE * file_encoded = NULL;
   FILE * file_FCIrec = NULL;
   FILE * file_mode = NULL;

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
   argv[3]=outFileName;
   argv[2]=inFileName;
   argv[1]="MR122";
   capacity=0;

   /*if((file_FCIrec = fopen(outFileName2, "w")) == NULL)
   {
	    printf("打开FCI记录失败！\n");
		return 1;
   }*/
   
   if ((argc == 5) || (argc == 4)){
      file_encoded = fopen(argv[argc - 1], "wb");
      if (file_encoded == NULL){
         Usage(argv);
		 printf("a\n");
         return 1;
      }
      file_speech = fopen(argv[argc - 2], "rb");
      if (file_speech == NULL){
         fclose(file_encoded);
         Usage(argv);
		 printf("b %s\n",argv[argc - 2]);
         return 1;
      }
      if (strncmp(argv[argc - 3], "-modefile=", 10) == 0){
         file_mode = fopen(&argv[argc - 3][10], "rt");
         if (file_mode == NULL){
            Usage(argv);
			printf("c\n");
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
            Usage(argv);
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
            Usage(argv);
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
      Usage(argv);
	  printf("f\n");
      return 1;
   }


   enstate = Encoder_Interface_init(dtx);

   //Copyright();
#ifndef VAD2
   fprintf( stderr, "%s\n", "Code compiled with VAD option: VAD1");
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
      byte_counter = Encoder_Interface_Encode(enstate, req_mode, speech, serial_data, 0,em_rate,enta);

      bytes += byte_counter;
      fwrite(serial_data, sizeof (UWord8), byte_counter, file_encoded );
      fflush(file_encoded);
   }
   Encoder_Interface_exit(enstate);

#ifndef ETSI
#ifdef IF2
   fprintf ( stderr, "\n%s%i%s%i%s\n", "Frame structure AMR IF2: ", frames, " frames, ", bytes, " bytes.");
#else
   fprintf ( stderr, "\n%s%i%s%i%s\n", "Frame structure AMR MIME file storage format: ", frames, " frames, ", bytes, " bytes.");
#endif
#else
   fprintf ( stderr, "\n%s%i%s\n", "Frame structure AMR ETSI: ", frames, " frames. ");
#endif

   //fprintf(file_FCIrec,"capacity=%d\n",capacity);
   fclose(file_speech);
   fclose(file_encoded);
   /*fclose(file_FCIrec);*/

   if (file_mode != NULL)
      fclose(file_mode);

   free(argv);

   return 0;
}
