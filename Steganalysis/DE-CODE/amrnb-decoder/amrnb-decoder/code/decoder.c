// decoder.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
/*
 * ===================================================================
 *  TS 26.104
 *  REL-5 V5.4.0 2004-03
 *  REL-6 V6.1.0 2004-03
 *  3GPP AMR Floating-point Speech Codec
 * ===================================================================
 *
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "interf_dec.h"
#include "sp_dec.h"
#include "typedef.h"

#ifndef ETSI
#ifndef IF2
#include <string.h>
#define AMR_MAGIC_NUMBER "#!AMR\n"
#endif
#endif



void Copyright(void){
fprintf (stderr,
"===================================================================\n"
" TS 26.104                                                         \n"
" REL-5 V5.4.0 2004-03                                              \n"
" REL-6 V6.1.0 2004-03                                              \n"
" 3GPP AMR Floating-point Speech Decoder                            \n"
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

int main()
{
	int filenu,em_rate;
	char pcm_path[100], dat_path[100], feafile[100], filename[100], time[100], Infile[100], Outfile[100], createfile[100];
	for (int time = 1; time <= 1;time++) {
		int total_frames = time * 50;//总帧数
		for (em_rate = 0; em_rate <= 100; em_rate += 10) {
			for (filenu = 0; filenu < 8000; filenu++)
			{
				sprintf(Infile, "D:\\paper1Data\\NB\\liu\\1s\\%d\\dat\\sample%04d.dat", em_rate,filenu);
				sprintf(Outfile, "D:\\paper1Data\\NB\\liu\\1s\\%d\\hiding\\hunag_sample%04d.pcm", em_rate, filenu);
				sprintf(feafile, "D:\\paper1Data\\NB\\liu\\1s\\%d\\alltxt\\AMRNB_sample%04d.txt",  em_rate,filenu);
				sprintf(createfile, "D:\\paper1Data\\NB\\liu\\1s");
				mkdirs(createfile);
				printf("---------------------------------------------\n%s\n%s\n%s\n", Infile, Outfile, feafile);
				decoder(Infile, Outfile, feafile,total_frames);//解码	

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
 *    Speech decoder main program
 *
 *    Usage: decoder bitstream_file synthesis_file
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
 *    Format for synthesis_file:
 *       Speech is written to a 16 bit 8kHz file.
 *
 *    ETSI bitstream file format is defined using ETSI as preprocessor
 *    definition
 * Returns:
 *    0
 */
int decoder(char *infilename,char *outfilename,char* feafile,int total_frames)
{
	int argc;
	char **argv=(char**)malloc(4*sizeof(char*));

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
   short block_size[16]={ 12, 13, 15, 17, 18, 20, 25, 30, 5, 0, 0, 0, 0, 0, 0, 0 };
#else
   char magic[8];
   short block_size[16]={ 12, 13, 15, 17, 19, 20, 26, 31, 5, 0, 0, 0, 0, 0, 0, 0 };
#endif
#else
   short analysis[250];
#endif
   
   argc=3;
   argv[1]=infilename;
   argv[2]=outfilename;

   /*argv[1]="C:\\Users\\Administrator\\Desktop\\sample000_o.dat";
   argv[2]="C:\\Users\\Administrator\\Desktop\\sample0000_o.pcm";*/

   /* Process command line options */
      if (argc == 3){

      file_speech = fopen(argv[2], "wb");
      if (file_speech == NULL){
         fprintf ( stderr, "%s%s%s\n","Use: ",argv[0], " no pcm file " );
         return 1;
      }

      file_analysis = fopen(argv[1], "rb");
      if (file_analysis == NULL){
         fprintf ( stderr, "%s%s%s\n","Use: ",argv[0], "no dat file " );
         fclose(file_speech);
         return 1;
      }

   }
   else {
      fprintf ( stderr, "%s%s%s\n","Use: ",argv[0], " input.file output.file " );
      return 1;
   }
   //Copyright();
   /* init decoder */
   destate = Decoder_Interface_init();

#ifndef ETSI
#ifndef IF2
   /* read and verify magic number */
   fread( magic, sizeof( char ), strlen( AMR_MAGIC_NUMBER ), file_analysis );
   if ( strncmp( magic, AMR_MAGIC_NUMBER, strlen( AMR_MAGIC_NUMBER ) ) ) {
	   fprintf( stderr, "%s%s\n", "Invalid magic number: ", magic );
	   fclose( file_speech );
	   fclose( file_analysis );
	   return 1;
   }
#endif
#endif

#ifndef ETSI

   /* find mode, read file */
   while (fread(analysis, sizeof (unsigned char), 1, file_analysis ) > 0)
   {
#ifdef IF2
      dec_mode = analysis[0] & 0x000F;
#else
      dec_mode = (analysis[0] >> 3) & 0x000F;
#endif
	  read_size = block_size[dec_mode];

      fread(&analysis[1], sizeof (char), read_size, file_analysis );
#else

   read_size = 250;
   /* read file */
   while (fread(analysis, sizeof (short), read_size, file_analysis ) > 0)
   {
#endif

      frames ++;

      /* call decoder */
      Decoder_Interface_Decode(destate, analysis, synth, 0, feafile,frames, total_frames);

      fwrite( synth, sizeof (short), 160, file_speech );
   }

   Decoder_Interface_exit(destate);

   fclose(file_speech);
   fclose(file_analysis);
   fprintf ( stderr, "%s%i%s\n","Decoded ", frames, " frames.");

   //for(i=0;i<2000;i++) printf("%d\n",lag_pitch[i]);

   return 0;
}
