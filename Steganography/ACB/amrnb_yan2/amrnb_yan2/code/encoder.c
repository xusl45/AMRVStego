// coder.cpp : 定义控制台应用程序的入口点。
//
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_NONSTDC_NO_DEPRECATE
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
#include "io.h"
#include "direct.h"
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
void mkdirs(char *muldir)
{
	int i, len;
	char str[512];
	strncpy(str, muldir, 512);
	len = strlen(str);
	for (i = 0; i < len; i++)
	{
		if (str[i] == '/')
		{
			str[i] = '\0';
			if (access(str, 0) != 0)
			{
				mkdir(str, 0777);
			}
			str[i] = '/';
		}
	}
	if (len > 0 && access(str, 0) != 0)
	{
		mkdir(str, 0777);
	}
	return;
}

void MakeRand(int arr[], int count)//打乱数组
{
	srand((unsigned int)time(NULL)); //随机数种子;
	for (int i = 0; i < count - 1; i++)
	{
		int num = i + rand() % (count - 1 - i); // 取随机数
		int temp = arr[i];
		arr[i] = arr[num];
		arr[num] = temp; //交换
	}
}
int main(int argc, char* argv[])
{
	int filenu, j; int em_rate;
	char Infile[100] = { 0 };
	char Outfile[100] = { 0 };
	char create_path[100], Outfile_c[100], stegofea[100];
	int emr_real;//实际嵌入率
	int *ste;
	//嵌入率=总的嵌入比特/总的嵌入容量(比特数)=总的嵌入比特/(总共嵌入帧数*每帧嵌入比特)
	//最大可嵌入比特数 = 嵌入率 * 总共嵌入帧数*每帧最大嵌入比特
	//char timee[100];
	//sprintf(timee, "%s", "0.9");
	//float time = strtod(timee, NULL);
	for (float time = 0.1; time < 0.99; time = time + 0.1)
	{
		int total_frames = time * 50;//总帧数
		int total_subframes = time * 50 * 4;//20ms一帧,1秒50帧,1帧有2整数(2,4)用于嵌入,可嵌入3bit秘密信息
		for (int em_rate = 0; em_rate <= 100; em_rate = em_rate + 100) {
			srand(8000);
			for (filenu = 0; filenu < 8000; filenu++)//样本个数
			{
				sprintf(Infile, "D:/paper1Data/NB/yan2/%.1fs/%d/hiding/c_sample%04d.pcm", time, em_rate, filenu);
				sprintf(Outfile, "D:/paper1Data/NB/yan2/%.1fs/%d/rendat/sample%04d.dat", time, em_rate, filenu);
				sprintf(Outfile_c, "D:/paper1Data/NB/yan2/%.1fs/%d/renhiding/c_sample%04d.pcm", time, em_rate, filenu);
				sprintf(stegofea, "D:/paper1Data/NB/yan2/%.1fs/%d/renall/AMRNB_sample%04d.txt", time, em_rate, filenu);
				sprintf(create_path, "D:/paper1Data/NB/yan2/%.1fs/%d/", time, em_rate);
				mkdirs(create_path);
				sprintf(create_path, "D:/paper1Data/NB/yan2/%.1fs/%d/", time, em_rate);
				mkdirs(create_path);
				//生成嵌入标识
				int ste_num = em_rate * total_frames / 100;//待嵌入的帧个数
				ste = (int*)malloc(total_frames * sizeof(int));
				for (int xx = 0; xx < ste_num; xx++)
					ste[xx] = 1;
				for (int xx = ste_num; xx < total_frames; xx++)
					ste[xx] = 0;
				MakeRand(ste, total_frames);
				//printf("嵌入标志:");
				//for (int xx = 0; xx < total_frames; xx++)
					//printf("%d ", ste[xx]);
				emr_real = 0;
				printf("---------------------%d/8000,time=%.1fs,emr=%d-------------------------\n%s\n%s\n%s\n%s\n", filenu, time, em_rate, Infile, Outfile_c, Outfile, stegofea);
				int change_frames = 0;//实际修改帧数
				coder(Infile, Outfile, em_rate, total_subframes, filenu, &change_frames, ste);
				emr_real = 100 * change_frames / total_frames;//实际嵌入率
				printf("嵌入率:%d,实际嵌入率:%d,改变帧数:%d/%d%d\n", em_rate, emr_real, change_frames, total_frames, ste);
				decoder(Outfile, Outfile_c, stegofea, total_frames);//解码
				free(ste);
			}
		}


	}

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
int coder (char *infilename, char *outfilename,int em_rate,int total_subframes,int filenu,int *change_frames,  int ste){
   
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
         Usage(argv);
		 //printf("a\n");
         return 1;
      }
      file_speech = fopen(argv[argc - 2], "rb");
      if (file_speech == NULL){
         fclose(file_encoded);
         Usage(argv);
		 //printf("b %s\n",argv[argc - 2]);
         return 1;
      }

      if (strncmp(argv[argc - 3], "-modefile=", 10) == 0){
         file_mode = fopen(&argv[argc - 3][10], "rt");
         if (file_mode == NULL){
            Usage(argv);
			//printf("kkk\n");
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
			//printf("d\n");
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
			//printf("e\n");
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
	  //printf("f\n");
      return 1;
   }


   enstate = Encoder_Interface_init(dtx);

   Copyright();
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
      byte_counter = Encoder_Interface_Encode(enstate, req_mode, speech, serial_data, 0, em_rate, total_subframes, filenu, change_frames, frames, ste);
	 
	  //byte_counter = Encoder_Interface_Encode(enstate, req_mode, speech, serial_data, 0);
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

   fclose(file_speech);
   fclose(file_encoded);
   if (file_mode != NULL)
      fclose(file_mode);

   return 0;
}
int decoder(char *infilename, char *outfilename, char* coverfile, int total_frames)
{
	int argc;
	char **argv = (char**)malloc(4 * sizeof(char*));

	//int main (int argc, char * argv[]){

	FILE * file_speech, *file_analysis;

	short synth[160];
	int frames = 0;
	int * destate;
	int read_size;
#ifndef ETSI
	unsigned char analysis[32];
	enum Mode dec_mode;
#ifdef IF2
	short block_size[16] = { 12, 13, 15, 17, 18, 20, 25, 30, 5, 0, 0, 0, 0, 0, 0, 0 };
#else
	char magic[8];
	short block_size[16] = { 12, 13, 15, 17, 19, 20, 26, 31, 5, 0, 0, 0, 0, 0, 0, 0 };
#endif
#else
	short analysis[250];
#endif

	argc = 3;
	argv[1] = infilename;
	argv[2] = outfilename;

	/*argv[1]="C:\\Users\\Administrator\\Desktop\\sample000_o.dat";
	argv[2]="C:\\Users\\Administrator\\Desktop\\sample0000_o.pcm";*/

	/* Process command line options */
	if (argc == 3) {

		file_speech = fopen(argv[2], "wb");
		if (file_speech == NULL) {
			fprintf(stderr, "%s%s%s\n", "Use: ", argv[0], " no pcm file ");
			return 1;
		}

		file_analysis = fopen(argv[1], "rb");
		if (file_analysis == NULL) {
			fprintf(stderr, "%s%s%s\n", "Use: ", argv[0], "no dat file ");
			fclose(file_speech);
			return 1;
		}

	}
	else {
		fprintf(stderr, "%s%s%s\n", "Use: ", argv[0], " input.file output.file ");
		return 1;
	}
	//Copyright();
	/* init decoder */
	destate = Decoder_Interface_init();

#ifndef ETSI
#ifndef IF2
	/* read and verify magic number */
	fread(magic, sizeof(char), strlen(AMR_MAGIC_NUMBER), file_analysis);
	if (strncmp(magic, AMR_MAGIC_NUMBER, strlen(AMR_MAGIC_NUMBER))) {
		fprintf(stderr, "%s%s\n", "Invalid magic number: ", magic);
		fclose(file_speech);
		fclose(file_analysis);
		return 1;
	}
#endif
#endif

#ifndef ETSI

	/* find mode, read file */
	while (fread(analysis, sizeof(unsigned char), 1, file_analysis) > 0)
	{
#ifdef IF2
		dec_mode = analysis[0] & 0x000F;
#else
		dec_mode = (analysis[0] >> 3) & 0x000F;
#endif
		read_size = block_size[dec_mode];

		fread(&analysis[1], sizeof(char), read_size, file_analysis);
#else

	read_size = 250;
	/* read file */
	while (fread(analysis, sizeof(short), read_size, file_analysis) > 0)
	{
#endif

		frames++;

		/* call decoder */
		Decoder_Interface_Decode(destate, analysis, synth, 0, coverfile, frames, total_frames);

		fwrite(synth, sizeof(short), 160, file_speech);
	}

	Decoder_Interface_exit(destate);

	fclose(file_speech);
	fclose(file_analysis);
	fprintf(stderr, "%s%i%s\n", "Decoded ", frames, " frames.");

	//for(i=0;i<2000;i++) ////printf("%d\n",lag_pitch[i]);

	return 0;
	}

